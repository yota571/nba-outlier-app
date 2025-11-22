# =====================================================
# NBA Prop Edge Finder – PrizePicks
# Version 2.0.0 (with ML integration)
# - Everything from v2.0.0 (bet tracker, moneylines, spreads, learning)
# - PLUS: true ML probability of Over (ml_prob_over, ml_odds_over)
#   using a trained scikit-learn model loaded from over_model.pkl
# =====================================================

import math
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import joblib  # ML model loader

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll

from DFS_Wrapper import PrizePick


# =====================================================
# STREAMLIT PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="NBA Outlier-Style App (PrizePicks, Cloud Version, ML)",
    layout="wide",
)

# Check for mobile-ish layout (just a simple heuristic for now)
MOBILE_WIDTH_THRESHOLD = 900
mobile_mode = st.session_state.get("browser_width", 1200) < MOBILE_WIDTH_THRESHOLD

st.title("🏀 NBA Outlier-Style App (PrizePicks, Cloud Version, ML)")
st.caption(
    "Full-game NBA props • PrizePicks lines vs nba_api stats • Edge, confidence, ML Over probability, and learning from bet history."
)

# =====================================================
# SESSION-LEVEL CONSTANTS / PATHS
# =====================================================
BETS_FILE = "bet_tracker.csv"  # persisted bet tracker
MODEL_FILE = "over_model.pkl"  # scikit-learn model for Over probability


# =====================================================
# HELPERS: PLAYER NAME NORMALIZATION
# =====================================================
@st.cache_data
def get_all_players():
    return players.get_players()


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    for ch in [".", ",", "'", "`"]:
        name = name.replace(ch, "")
    name = " ".join(name.split())
    return name


@st.cache_data
def get_player_id(player_name: str):
    all_players = get_all_players()
    if not player_name:
        return None

    target = normalize_name(player_name)

    # exact match
    for p in all_players:
        if normalize_name(p["full_name"]) == target:
            return p["id"]

    # contains match
    for p in all_players:
        norm = normalize_name(p["full_name"])
        if target and (target in norm or norm in target):
            return p["id"]

    # initials like "K. Caldwell-Pope"
    parts = target.split()
    if len(parts) > 1 and len(parts[0]) == 1:
        target_no_initial = " ".join(parts[1:])
        for p in all_players:
            norm = normalize_name(p["full_name"])
            if target_no_initial and (target_no_initial in norm or norm in target_no_initial):
                return p["id"]

    # last-name-only unique match
    if len(parts) >= 2:
        last_name = parts[-1]
        candidates = []
        for p in all_players:
            norm = normalize_name(p["full_name"])
            if last_name and last_name in norm:
                candidates.append(p["id"])
        if len(candidates) == 1:
            return candidates[0]

    return None


@st.cache_data
def get_player_gamelog(player_id: int) -> pd.DataFrame:
    if player_id is None:
        return pd.DataFrame()
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=SeasonAll.all)
        df = gl.get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        df = df.sort_values("GAME_DATE", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


# =====================================================
# MARKET SERIES MAPPING
# =====================================================
def get_market_series(gamelog_df: pd.DataFrame, market: str) -> pd.Series:
    market = (market or "").lower().strip()
    if gamelog_df.empty:
        return pd.Series(dtype="float")

    for col in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]:
        if col not in gamelog_df.columns:
            gamelog_df[col] = 0

    if market == "points":
        return gamelog_df["PTS"]
    if market == "rebounds":
        return gamelog_df["REB"]
    if market == "assists":
        return gamelog_df["AST"]
    if market == "pra":
        return gamelog_df["PTS"] + gamelog_df["REB"] + gamelog_df["AST"]
    if market == "ra":
        return gamelog_df["REB"] + gamelog_df["AST"]
    if market == "threes":
        return gamelog_df["FG3M"]
    if market == "fs":
        # Fantasy Score: a simple FP model (can be adjusted to your site’s scoring)
        return (
            gamelog_df["PTS"]
            + 1.2 * gamelog_df["REB"]
            + 1.5 * gamelog_df["AST"]
            + 3.0 * (gamelog_df["STL"] + gamelog_df["BLK"])
            - gamelog_df["TOV"]
        )

    return pd.Series(dtype="float")


# =====================================================
# HELPERS: FILTER FULL-GAME PROPS ONLY (PrizePicks data)
# =====================================================
def _is_full_game_prop_pp(item: dict, stat_type: str | None) -> bool:
    """
    PrizePicks: detect and exclude 1H / 2H / quarter / first 5-min props.
    We only want full-game props so we can use full-game stats.
    """
    text_bits = []

    for key in [
        "description",
        "short_description",
        "label",
        "title",
        "stat_type",
        "market_type",
        "game_type",
        "league",
    ]:
        v = item.get(key)
        if v:
            text_bits.append(str(v).lower())

    joined = " ".join(text_bits)

    bad_keywords = [
        "1h",
        " 1h ",
        "2h",
        "first half",
        "1st half",
        "second half",
        "2nd half",
        "half points",
        "half pts",
        "1h points",
        "1h pts",
        "1st quarter",
        "2nd quarter",
        "3rd quarter",
        "4th quarter",
        "first quarter",
        "second quarter",
        "third quarter",
        "fourth quarter",
        "q1",
        "q2",
        "q3",
        "q4",
        "first 5",
        "in first 5",
        "first five",
        "first 3 min",
        "first 6 min",
        "first 7 min",
        "in first six",
        "in first seven",
    ]

    for kw in bad_keywords:
        if kw in joined:
            return False

    period = str(item.get("period", "")).lower()
    if period and period not in ("", "full", "full game", "game"):
        return False

    scope = str(item.get("scope", "")).lower()
    if any(x in scope for x in ["half", "quarter", "1h", "2h", "q1", "q2", "q3", "q4"]):
        return False

    return True


# =====================================================
# PRIZEPICKS LOADER (NBA, FULL GAME ONLY)
# =====================================================
@st.cache_data
def load_prizepicks_nba_props() -> pd.DataFrame:
    """
    Pull cleaned NBA props from PrizePicks.

    - NBA only
    - pre-game only
    - full-game only (no halves/quarters/first 5 min)
    - keep ONLY one line per (player, team, opponent, market) – highest line
    """
    try:
        pp = PrizePick()
        raw = pp.get_data(organize_data=False)
    except TypeError:
        # older / different signature fallback
        pp = PrizePick()
        raw = pp.get_data(False)
    except Exception as e:
        st.error(f"Error loading PrizePicks data: {e}")
        return pd.DataFrame(
            columns=[
                "player_name",
                "team",
                "opponent",
                "market",
                "line",
                "game_time",
                "book",
            ]
        )

    if not isinstance(raw, list):
        st.warning("PrizePicks data not in expected list format.")
        return pd.DataFrame(
            columns=[
                "player_name",
                "team",
                "opponent",
                "market",
                "line",
                "game_time",
                "book",
            ]
        )

    records = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        league = item.get("league", "")
        if "NBA" not in str(league).upper():
            continue

        status = str(item.get("status", "")).lower()
        if status and status not in ("pre_game", "pre-game", "pregame"):
            continue

        stat_type = item.get("stat_type")

        if not _is_full_game_prop_pp(item, stat_type):
            continue

        line_val = item.get("line_score")
        if line_val is None:
            line_val = item.get("line_value")
        if line_val is None:
            continue

        try:
            line_val = float(line_val)
        except Exception:
            continue

        player_name = item.get("player_name") or item.get("name")
        if not player_name:
            continue

        team = item.get("team")
        opponent = item.get("opponent") or ""
        start_time = item.get("start_time") or item.get("game_date_time")

        market_map = {
            "Points": "points",
            "Rebounds": "rebounds",
            "Assists": "assists",
            "Pts + Rebs + Asts": "pra",
            "Pts+Rebs+Asts": "pra",
            "Points + Rebounds + Assists": "pra",
            "Rebs+Asts": "ra",
            "Rebs + Asts": "ra",
            "Rebounds + Assists": "ra",
            "3-Pointers Made": "threes",
            "Fantasy Score": "fs",
            "Fantasy Points": "fs",
        }
        market = market_map.get(stat_type)
        if market is None:
            continue

        # skip combos etc.
        if "+" in player_name:
            continue

        # skip split opponents
        if any(sep in opponent for sep in ("/", "|", "+")):
            continue

        # basic sanity filters so junk lines don't clutter
        if market == "points" and line_val < 10:
            continue
        if market == "rebounds" and line_val < 3:
            continue
        if market == "assists" and line_val < 3:
            continue
        if market == "pra" and line_val < 15:
            continue
        if market == "ra" and line_val < 6:
            continue
        if market == "fs" and line_val < 15:
            continue

        records.append(
            {
                "player_name": player_name,
                "team": team,
                "opponent": opponent,
                "market": market,
                "line": line_val,
                "game_time": start_time,
                "book": "PrizePicks",
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "player_name",
                "team",
                "opponent",
                "market",
                "line",
                "game_time",
                "book",
            ]
        )

    df = pd.DataFrame(records)
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df.dropna(subset=["line"])
    df = df.sort_values("line")

    df = df.groupby(
        ["player_name", "team", "opponent", "market"],
        as_index=False,
    ).last()

    df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df[
        ["player_name", "team", "opponent", "market", "line", "game_time", "book"]
    ]


# =====================================================
# CSV LOADER (OPTIONAL MANUAL PROPS)
# =====================================================
@st.cache_data
def load_props_from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    required_cols = ["player_name", "team", "opponent", "market", "line", "game_time"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")
    if "book" not in df.columns:
        df["book"] = "Custom"

    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df.dropna(subset=["line"])

    return df[required_cols + ["book"]]


# =====================================================
# ML MODEL LOADING (Over probability)
# =====================================================
@st.cache_resource
def load_over_model():
    if not os.path.exists(MODEL_FILE):
        st.warning(
            f"ML model file `{MODEL_FILE}` not found. "
            "ML Over probabilities will be disabled."
        )
        return None

    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading ML model from `{MODEL_FILE}`: {e}")
        return None


over_model = load_over_model()


def compute_ml_over_prob(
    line: float,
    market: str,
    season_avg: float,
    avg_last_n: float,
    avg_last_7: float | None,
    over_rate_n: float,
    over_rate_7: float | None,
    edge_n: float,
    edge_7: float | None,
) -> tuple[float | None, float | None]:
    """
    Use scikit-learn model to predict probability of Over.
    Returns (prob_over, decimal_odds_over) or (None, None) if model not available.
    """
    if over_model is None:
        return None, None

    try:
        # Simple feature vector – you can expand this to match your trained model
        feats = {
            "line": float(line),
            "season_avg": float(season_avg),
            "avg_last_n": float(avg_last_n),
            "avg_last_7": float(avg_last_7 if avg_last_7 is not None else avg_last_n),
            "over_rate_n": float(over_rate_n),
            "over_rate_7": float(over_rate_7 if over_rate_7 is not None else over_rate_n),
            "edge_n": float(edge_n),
            "edge_7": float(edge_7 if edge_7 is not None else edge_n),
            "is_points": 1.0 if market == "points" else 0.0,
            "is_rebounds": 1.0 if market == "rebounds" else 0.0,
            "is_assists": 1.0 if market == "assists" else 0.0,
            "is_pra": 1.0 if market == "pra" else 0.0,
            "is_ra": 1.0 if market == "ra" else 0.0,
            "is_threes": 1.0 if market == "threes" else 0.0,
            "is_fs": 1.0 if market == "fs" else 0.0,
        }

        X = pd.DataFrame([feats])
        prob_over = float(over_model.predict_proba(X)[0, 1])  # prob of class "Over"
        prob_over = max(0.0001, min(0.9999, prob_over))  # clamp extremes

        # Convert to fair decimal odds
        ml_odds_over = 1.0 / prob_over if prob_over > 0 else None
        return prob_over, ml_odds_over
    except Exception:
        return None, None


# =====================================================
# SIDEBAR UI
# =====================================================
st.sidebar.title("NBA Outlier-Style App (PrizePicks, Cloud, ML)")

mode = st.sidebar.radio(
    "Prop source",
    [
        "PrizePicks (live)",
        "Upload CSV manually",
    ],
)

games_to_look_back = st.sidebar.slider(
    "Games to look back (N)", min_value=5, max_value=25, value=10, step=1
)

min_over_rate = st.sidebar.slider(
    "Minimum Over % (last N games)", min_value=0.0, max_value=1.0, value=0.6, step=0.05
)

min_edge = st.sidebar.number_input("Minimum Edge (Avg - Line)", value=1.0, step=0.5)

min_confidence = st.sidebar.slider(
    "Minimum Confidence %", min_value=0, max_value=100, value=60, step=5
)

only_today = st.sidebar.checkbox("Only today's games (by game_time)", value=False)

st.sidebar.markdown("---")
with st.sidebar.expander("What is Edge / Confidence / Bet side?"):
    st.write(
        "- **Edge** = how much higher the player has been performing vs the line (last N games).\n"
        "- **Confidence %** = blends hit-rate and edge size; higher = stronger data signal.\n"
        "- **Bet side** = Over / Under / No clear edge, based on predicted score vs the line.\n"
        "- This app uses **full-game stats vs full-game PrizePicks props only.**"
    )

st.sidebar.markdown(
    "Markets used: **points, rebounds, assists, pra, ra (reb+ast), threes, fs (fantasy)**"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "ML Over probabilities are from a local scikit-learn model (`over_model.pkl`)."
)


# =====================================================
# LOAD PROPS
# =====================================================
props_df = None

if mode == "PrizePicks (live)":
    props_df = load_prizepicks_nba_props()
elif mode == "Upload CSV manually":
    uploaded = st.sidebar.file_uploader(
        "Upload props CSV",
        type=["csv"],
        help="Must contain: player_name, team, opponent, market, line, game_time (optional: book)",
    )
    if uploaded:
        props_df = load_props_from_csv(uploaded)

if props_df is None or props_df.empty:
    st.info("No props loaded yet.")
    st.stop()

if only_today and "game_time" in props_df.columns:
    today = datetime.today().date()
    has_time = props_df["game_time"].notna()
    today_mask = has_time & (props_df["game_time"].dt.date == today)
    no_time_mask = ~has_time
    props_df = pd.concat(
        [props_df[today_mask], props_df[no_time_mask]],
        ignore_index=True,
    )

if props_df.empty:
    st.warning("No props after applying date filter.")
    st.stop()

# Top filters
teams = ["All"] + sorted(props_df["team"].dropna().unique().tolist())
markets = ["All"] + sorted(props_df["market"].dropna().unique().tolist())

top_col1, top_col2, top_col3 = st.columns([2, 2, 3])
with top_col1:
    team_filter = st.selectbox("Team filter", teams)
with top_col2:
    market_filter = st.selectbox("Market filter", markets)
with top_col3:
    search_name = st.text_input("Search player (optional)", "")

df = props_df.copy()

if team_filter != "All":
    df = df[df["team"] == team_filter]
if market_filter != "All":
    df = df[df["market"].str.lower() == market_filter.lower()]

if df.empty:
    st.warning("No props match the selected filters.")
    st.stop()


# =====================================================
# EDGE / CONFIDENCE / PREDICTION / BET SIDE + ML
# =====================================================
st.write("### Calculating edges & ML probabilities…")

rows = []
errors = []

unique_players = sorted(df["player_name"].dropna().unique().tolist())
player_logs = {}
player_ids = {}

progress = st.progress(0.0)
status_text = st.empty()

total_players = len(unique_players) if unique_players else 1

for i, name in enumerate(unique_players):
    status_text.text(f"Fetching NBA stats for players… ({i+1}/{total_players})")

    pid = get_player_id(name)
    player_ids[name] = pid
    if pid is None:
        errors.append(f"Player not found in nba_api: {name}")
        continue

    glog = get_player_gamelog(pid)
    if glog.empty:
        errors.append(f"No game log for player: {name}")
        continue

    player_logs[name] = glog

    time.sleep(0.2)  # be gentle to NBA API
    progress.progress((i + 1) / total_players)

progress.progress(1.0)
status_text.text("Computing edges, predictions, ML probabilities, confidence, bet side…")

for _, row in df.iterrows():
    player_name = row.get("player_name")
    market = row.get("market")
    line = row.get("line")
    book = row.get("book") or "PrizePicks"

    if player_name is None or market is None or line is None:
        continue
    if player_name not in player_logs:
        continue

    try:
        line_float = float(line)
    except Exception:
        errors.append(f"Invalid line for {player_name}: {line}")
        continue

    gamelog = player_logs[player_name]
    series = get_market_series(gamelog, market).dropna()
    if series.empty:
        errors.append(f"No stats for {player_name} – market '{market}'")
        continue

    season_avg = series.mean()

    last_n = series.iloc[:games_to_look_back]
    if last_n.empty:
        errors.append(f"No recent games for {player_name}")
        continue

    avg_last_n = last_n.mean()
    over_rate_n = (last_n > line_float).mean()
    edge_n = avg_last_n - line_float

    last7 = series.iloc[:7]
    if len(last7) > 0:
        avg_last_7 = last7.mean()
        over_rate_7 = (last7 > line_float).mean()
        edge_7 = avg_last_7 - line_float
    else:
        avg_last_7 = None
        over_rate_7 = None
        edge_7 = None

    # blended prediction (non-ML baseline)
    w_season = 0.4
    w_last_n = 0.4
    w_last_7 = 0.2

    avg7_for_blend = avg_last_7 if avg_last_7 is not None else avg_last_n
    predicted_score = (
        w_season * season_avg
        + w_last_n * avg_last_n
        + w_last_7 * avg7_for_blend
    )

    # ML Over probability
    ml_prob_over, ml_odds_over = compute_ml_over_prob(
        line_float,
        market,
        season_avg,
        avg_last_n,
        avg_last_7,
        over_rate_n,
        over_rate_7,
        edge_n,
        edge_7,
    )

    # base confidence pieces
    hit_score = over_rate_n

    if line_float != 0:
        edge_ratio = max(0.0, edge_n / max(1.0, line_float))
    else:
        edge_ratio = 0.0
    edge_score = max(0.0, min(1.0, edge_ratio * 4.0))

    # blend in ML Over probability if available
    if ml_prob_over is not None:
        ml_score = 2.0 * abs(ml_prob_over - 0.5)  # 0.5=coin flip, 1.0=lock
        confidence = 0.45 * hit_score + 0.35 * edge_score + 0.20 * ml_score
    else:
        confidence = 0.6 * hit_score + 0.4 * edge_score

    confidence_pct = round(confidence * 100, 1)

    # predicted side from blended prediction
    delta = predicted_score - line_float
    if delta >= 0.5:
        bet_side = "Over"
    elif delta <= -0.5:
        bet_side = "Under"
    else:
        bet_side = "No clear edge"

    rows.append(
        {
            "player": player_name,
            "player_id": player_ids.get(player_name),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "market": market,
            "line": line_float,
            "book": book,
            "bet_side": bet_side,
            "season_avg": round(season_avg, 2),
            f"avg_last_{games_to_look_back}": round(avg_last_n, 2),
            f"over_rate_last_{games_to_look_back}": round(over_rate_n, 2),
            f"edge_last_{games_to_look_back}": round(edge_n, 2),
            "avg_last_7": round(avg_last_7, 2) if avg_last_7 is not None else None,
            "over_rate_last_7": round(over_rate_7, 2) if over_rate_7 is not None else None,
            "edge_last_7": round(edge_7, 2) if edge_7 is not None else None,
            "predicted_score": round(predicted_score, 2),
            "confidence_pct": confidence_pct,
            "ml_prob_over": round(ml_prob_over * 100, 1) if ml_prob_over is not None else None,
            "ml_odds_over": round(ml_odds_over, 2) if ml_odds_over is not None else None,
            "game_time": row.get("game_time"),
        }
    )

if not rows:
    st.error("No edges could be calculated from the current props.")
    if errors:
        with st.expander("Warnings/errors (players/markets skipped)"):
            for e in errors:
                st.write("- ", e)
    st.stop()

edges_df = pd.DataFrame(rows)

edge_cols = [
    c
    for c in edges_df.columns
    if c.startswith("edge_last_") and "7" not in c
]
rate_cols = [
    c
    for c in edges_df.columns
    if c.startswith("over_rate_last_") and "7" not in c
]

if not edge_cols or not rate_cols:
    st.error("Edge/over_rate columns missing from results.")
    st.stop()

edge_col = edge_cols[0]
rate_col = rate_cols[0]

# Apply player search filter (for card + table views)
if search_name.strip():
    view_df = edges_df[
        edges_df["player"].str.contains(search_name, case=False, na=False)
    ]
else:
    view_df = edges_df.copy()

if view_df.empty:
    st.warning("No props after applying search filter.")
    st.stop()


# =====================================================
# BET TRACKER PERSISTENCE
# =====================================================
@st.cache_data
def _load_bets_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_bets_from_disk() -> list[dict]:
    df = _load_bets_file(BETS_FILE)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def save_bets_to_disk(bets: list[dict]):
    df = pd.DataFrame(bets)
    try:
        df.to_csv(BETS_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving bet tracker to disk: {e}")


if "bet_tracker" not in st.session_state:
    st.session_state["bet_tracker"] = load_bets_from_disk()


def track_bet(row: pd.Series, ml: bool = False, spread: bool = False):
    """Append a bet to session + disk."""
    bet_type = "PROP"
    if ml:
        bet_type = "ML"
    elif spread:
        bet_type = "SPREAD"

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "bet_type": bet_type,
        "player": row.get("player"),
        "team": row.get("team"),
        "opponent": row.get("opponent"),
        "market": row.get("market"),
        "line": row.get("line"),
        "book": row.get("book"),
        "bet_side": row.get("bet_side"),
        "predicted_score": row.get("predicted_score"),
        "confidence_pct": row.get("confidence_pct"),
        "ml_prob_over": row.get("ml_prob_over"),
        "ml_odds_over": row.get("ml_odds_over"),
        "game_time": row.get("game_time"),
        "result": None,  # "Hit" / "Miss" / "Push"
    }

    st.session_state["bet_tracker"].append(record)
    save_bets_to_disk(st.session_state["bet_tracker"])
    st.success("Bet tracked!")


# =====================================================
# TABS: CARDS + EXPLANATION / TABLE / PLAYER DETAIL / GAMES / BET TRACKER
# =====================================================
tab_cards, tab_table, tab_player, tab_games, tab_bets = st.tabs(
    ["Cards & Explanation", "Table", "Player Detail", "Games (ML & Spreads)", "Bet Tracker"]
)


# =====================================================
# TAB 1: CARD VIEW (WITH HEADSHOTS) + “WHY THIS PROP?”
# =====================================================
with tab_cards:
    st.write("### Featured Edges (Card View)")

    filtered_edges = view_df[
        (view_df[rate_col] >= min_over_rate)
        & (view_df[edge_col] >= min_edge)
        & (view_df["confidence_pct"] >= min_confidence)
    ]

    if filtered_edges.empty:
        featured_df = view_df.copy()
        st.caption("No props match all filters yet – showing best available edges instead.")
    else:
        featured_df = filtered_edges.copy()

    featured_df = featured_df.sort_values(
        by=["confidence_pct", rate_col, edge_col], ascending=False
    ).reset_index(drop=True)

    top_n = min(12, len(featured_df))
    if top_n == 0:
        st.info("No edges available to display.")
    else:
        for idx in range(0, top_n, 2):
            cols = st.columns(2)
            for j in range(2):
                k = idx + j
                if k >= top_n:
                    break
                r = featured_df.iloc[k]
                with cols[j]:
                    card = st.container()
                    with card:
                        # top row: image + basic info
                        top_col1, top_col2 = st.columns([1, 2])

                        with top_col1:
                            pid = r.get("player_id")
                            if pid:
                                try:
                                    img_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{int(pid)}.png"
                                    st.image(img_url, use_column_width=True)
                                except Exception:
                                    st.write(" ")
                            else:
                                st.write(" ")

                        with top_col2:
                            st.markdown(f"### {r['player']}")
                            team = r.get("team") or ""
                            opp = r.get("opponent") or ""
                            st.markdown(f"**{team} vs {opp}**")
                            st.markdown(
                                f"*{r['market']}* &nbsp; | &nbsp; **Line:** `{r['line']}`"
                            )

                            side = r.get("bet_side", "No clear edge")
                            side_emoji = (
                                "⬆️" if side == "Over"
                                else "⬇️" if side == "Under"
                                else "⚖️"
                            )
                            st.markdown(f"**Recommended:** {side_emoji} **{side}**")

                            if r.get("ml_prob_over") is not None:
                                st.markdown(
                                    f"ML Over prob: **{r['ml_prob_over']:.1f}%**  "
                                    f"(Fair odds: {r['ml_odds_over']})"
                                )

                            st.markdown(
                                f"Predicted: **{r['predicted_score']}**  "
                                f"(Season avg: {r['season_avg']})"
                            )

                        # bottom row: metrics + confidence bar
                        conf = r.get("confidence_pct", 0) or 0
                        edge_val = r.get(edge_col, 0) or 0
                        hit = r.get(rate_col, 0) or 0

                        st.markdown(
                            f"Edge: `{edge_val:.2f}` &nbsp; | &nbsp; "
                            f"Hit rate (N): `{hit:.2f}` &nbsp; | &nbsp; "
                            f"Confidence: `{conf:.1f}%`"
                        )
                        st.progress(min(max(conf / 100.0, 0.0), 1.0))

                        # Track bet button
                        if st.button(
                            "Track this bet",
                            key=f"track_{r['player']}_{r['market']}_{k}",
                        ):
                            track_bet(r)

                        # Explanation button
                        if st.button(
                            "Why this prop?",
                            key=f"why_{r['player']}_{r['market']}_{k}",
                        ):
                            st.session_state["explain_row"] = r.to_dict()

    st.write("### Prop Explanation (Why this prediction & confidence)")

    if "explain_row" in st.session_state:
        er = st.session_state["explain_row"]

        player_name = er.get("player")
        market = er.get("market")
        line = er.get("line")

        st.markdown(
            f"**Player:** {player_name}  \n"
            f"**Market:** {market}  \n"
            f"**Line:** `{line}`"
        )

        mask = (
            (edges_df["player"] == player_name)
            & (edges_df["market"] == market)
            & (edges_df["line"] == line)
        )
        row_matches = edges_df[mask]
        if row_matches.empty:
            st.warning("Could not find full data for this prop in the edges table.")
        else:
            r = row_matches.iloc[0]

            season_avg = r.get("season_avg")
            avg_last_n = r.get(f"avg_last_{games_to_look_back}")
            over_rate_n = r.get(f"over_rate_last_{games_to_look_back}")
            edge_n = r.get(f"edge_last_{games_to_look_back}")
            avg_last_7 = r.get("avg_last_7")
            over_rate_7 = r.get("over_rate_last_7")
            edge_7 = r.get("edge_last_7")
            predicted_score = r.get("predicted_score")
            confidence_pct = r.get("confidence_pct")
            bet_side = r.get("bet_side")
            ml_prob_over = r.get("ml_prob_over")
            ml_odds_over = r.get("ml_odds_over")

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Key Numbers**")
                st.write(f"- Season average: **{season_avg}**")
                st.write(f"- Last {games_to_look_back} avg: **{avg_last_n}**")
                st.write(f"- Hit rate last {games_to_look_back}: **{over_rate_n:.2f}**")
                st.write(
                    f"- Edge last {games_to_look_back}: **{edge_n:.2f}** vs line **{line}**"
                )
                if avg_last_7 is not None:
                    st.write(f"- Last 7 avg: **{avg_last_7}**")
                    st.write(f"- Hit rate last 7: **{(over_rate_7 or 0):.2f}**")
                    st.write(f"- Edge last 7: **{(edge_7 or 0):.2f}**")

            with col_right:
                st.markdown("**Model Result**")
                st.write(f"- Predicted score: **{predicted_score}**")
                st.write(f"- Confidence: **{confidence_pct:.1f}%**")
                st.write(f"- Recommended side: **{bet_side}**")
                if ml_prob_over is not None:
                    st.write(
                        f"- ML Over probability: **{ml_prob_over:.1f}%** "
                        f"(fair odds: **{ml_odds_over}**)"
                    )

                st.markdown("**How confidence is calculated:**")
                st.write(
                    "- Take the hit rate over the last N games (how often they went over the line).\n"
                    "- Measure how far the average is **above** the line (edge).\n"
                    "- Blend with ML Over probability when available.\n"
                    "- Combine them: `Confidence = 0.45 * HitRate + 0.35 * Edge + 0.20 * MLScore`."
                )

            # Game-by-game history
            if player_name in player_logs:
                gamelog = player_logs[player_name]
            else:
                pid = get_player_id(player_name)
                gamelog = get_player_gamelog(pid)

            series = get_market_series(gamelog, market).dropna()
            last_n_vals = series.iloc[:games_to_look_back]

            if not last_n_vals.empty:
                hist_df = pd.DataFrame(
                    {
                        "GAME_DATE": gamelog["GAME_DATE"]
                        .iloc[:games_to_look_back]
                        .values,
                        market.upper(): last_n_vals.values,
                        "vs_line": [
                            "Over" if v > line else "Under" if v < line else "Push"
                            for v in last_n_vals.values
                        ],
                    }
                )
                st.markdown(
                    f"**Last {games_to_look_back} games for {market.upper()}**"
                )
                st.dataframe(hist_df, use_container_width=True)
            else:
                st.info("Not enough recent games to show game-by-game history.")
    else:
        st.caption(
            "Click **“Why this prop?”** on any card above to see a detailed explanation here."
        )


# =====================================================
# TAB 2: FULL EDGES TABLE
# =====================================================
with tab_table:
    st.write("### Full Edges Table")

    def highlight_edges(row):
        styles = [""] * len(row)
        try:
            val = float(row[edge_col])
            over = float(row[rate_col])
            conf = float(row["confidence_pct"])
            side = str(row["bet_side"])
        except Exception:
            return styles

        if (
            val >= min_edge
            and over >= min_over_rate
            and conf >= min_confidence
            and side in ("Over", "Under")
        ):
            color = "background-color: #d4f8d4"  # green for strong edges
        elif val < 0:
            color = "background-color: #f8d4d4"  # red if underperforming line
        elif val > 0:
            color = "background-color: #fff4c2"  # yellow if mild positive
        else:
            return styles

        return [color] * len(row)

    display_cols = [
        "player",
        "team",
        "opponent",
        "market",
        "line",
        "book",
        "bet_side",
        "predicted_score",
        "season_avg",
        f"avg_last_{games_to_look_back}",
        f"over_rate_last_{games_to_look_back}",
        f"edge_last_{games_to_look_back}",
        "avg_last_7",
        "over_rate_last_7",
        "edge_last_7",
        "confidence_pct",
        "ml_prob_over",
        "ml_odds_over",
        "game_time",
    ]

    table_df = view_df.copy()  # use search-filtered df
    for col in display_cols:
        if col not in table_df.columns:
            table_df[col] = None

    styled_edges = (
        table_df[display_cols]
        .sort_values(by=["confidence_pct", rate_col, edge_col], ascending=False)
        .style.apply(highlight_edges, axis=1)
    )

    st.dataframe(styled_edges, use_container_width=True)

    if errors:
        with st.expander("Warnings/errors (players/markets skipped)"):
            st.caption(
                "These players either aren't in nba_api yet or have no game log data. "
                "They are skipped but do not break the app."
            )
            for e in errors:
                st.write("- ", e)


# =====================================================
# TAB 3: PLAYER DETAIL (RAW GAME LOG)
# =====================================================
with tab_player:
    st.write("### Player Detail (Raw Game Log)")

    player_options = [""] + sorted(edges_df["player"].unique().tolist())
    selected_player = st.selectbox("Select a player", player_options)

    if selected_player:
        if selected_player in player_logs:
            gamelog = player_logs[selected_player]
        else:
            pid = get_player_id(selected_player)
            gamelog = get_player_gamelog(pid)

        if not gamelog.empty:
            st.write(f"Recent games for **{selected_player}**")
            st.dataframe(
                gamelog.head(games_to_look_back), use_container_width=True
            )

            # Allow user-selected market + line here to recalc quickly
            st.markdown("#### Quick Line Adjuster")
            mkt = st.selectbox(
                "Market",
                ["points", "rebounds", "assists", "pra", "ra", "threes", "fs"],
                index=0,
            )
            user_line = st.number_input(
                "Your custom line", value=float(gamelog["PTS"].mean() or 20.0)
            )

            series = get_market_series(gamelog, mkt).dropna()
            if not series.empty:
                last_n = series.iloc[:games_to_look_back]
                avg_last_n = last_n.mean()
                over_rate_n = (last_n > user_line).mean()
                edge_n = avg_last_n - user_line

                st.write(
                    f"- Avg last {games_to_look_back}: **{avg_last_n:.2f}**  "
                    f"(hit rate vs your line: **{over_rate_n:.2f}**)"
                )
                st.write(f"- Edge vs your line: **{edge_n:.2f}**")

            else:
                st.info("No data for this market for this player.")
        else:
            st.warning("No game log available for this player.")


# =====================================================
# TAB 4: GAME MONEYLINES & SPREADS (MODEL)
# =====================================================
with tab_games:
    st.write("### Game Moneylines & Model Spreads")

    if view_df.empty:
        st.info(
            "Not enough scoring-related props to build game-level moneyline/spread predictions yet.\n\n"
            "Add more props, especially points-related ones, then re-run."
        )
    else:
        # Build a simple game-level model:
        # we treat each prop as a "signal" for which team will win,
        # weighted by confidence, ML Over prob, and edge.

        game_rows = []

        for _, r in view_df.iterrows():
            team = r.get("team")
            opp = r.get("opponent")
            game_time = r.get("game_time")

            if not team or not opp:
                continue

            home = sorted([team, opp])[0]
            away = sorted([team, opp])[1]
            game_key = f"{home}_vs_{away}"

            conf = r.get("confidence_pct") or 0.0
            edge_val = r.get(edge_col) or 0.0
            prob_over = (r.get("ml_prob_over") or 50.0) / 100.0

            signal = (conf / 100.0) * edge_val * (prob_over - 0.5)

            game_rows.append(
                {
                    "game_key": game_key,
                    "game_label": f"{team} vs {opp}",
                    "team": team,
                    "opponent": opp,
                    "signal": signal,
                    "confidence_pct": conf,
                    "edge_val": edge_val,
                    "prob_over": prob_over,
                    "game_time": game_time,
                }
            )

        if not game_rows:
            st.info("No usable props to build game predictions.")
        else:
            games_df = pd.DataFrame(game_rows)
            if games_df.empty:
                st.info("No usable props to build game predictions.")
            else:
                # Aggregate by team within each game
                agg = (
                    games_df.groupby(["game_key", "team", "opponent"], as_index=False)
                    .agg(
                        avg_signal=("signal", "mean"),
                        avg_confidence=("confidence_pct", "mean"),
                        avg_edge=("edge_val", "mean"),
                        props_count=("signal", "size"),
                        avg_prob_over=("prob_over", "mean"),
                        game_time=("game_time", "first"),
                    )
                )

                # Turn avg_signal into a "win probability" inside each game
                def normalize_game_probs(grp):
                    """
                    Convert avg_signal into a 0–1 win probability per team
                    within each game, then derive a model spread.
                    """
                    if grp.empty:
                        return grp

                    def to_prob(sig):
                        try:
                            return 1.0 / (1.0 + math.exp(-0.8 * sig))
                        except OverflowError:
                            return 1.0 if sig > 0 else 0.0

                    grp["raw_prob"] = grp["avg_signal"].apply(to_prob)
                    grp["sum_prob_game"] = grp.groupby("game_key")["raw_prob"].transform(
                        "sum"
                    )

                    def norm_prob(row):
                        s = row["sum_prob_game"]
                        rp = row["raw_prob"]
                        if s <= 0:
                            return 0.5
                        return rp / s

                    grp["win_prob"] = grp.apply(norm_prob, axis=1)

                    # Convert win_prob to an approximate spread
                    # (very rough: 12 points = 90% vs 10%)
                    grp["model_spread"] = (grp["win_prob"] - 0.5) * 24.0
                    return grp

                ml_df = normalize_game_probs(agg)

                # Turn into favorite/underdog
                games_ml_df = ml_df.copy()
                games_ml_df["win_prob_pct"] = games_ml_df["win_prob"] * 100.0
                games_ml_df["ml_odds"] = games_ml_df["win_prob"].apply(
                    lambda p: 1.0 / max(p, 0.0001)
                )

                st.markdown("#### Model Moneylines (team-level)")

                ml_cols = [
                    "game_key",
                    "team",
                    "opponent",
                    "props_count",
                    "avg_confidence",
                    "avg_edge",
                    "win_prob_pct",
                    "ml_odds",
                    "game_time",
                ]
                for c in ml_cols:
                    if c not in games_ml_df.columns:
                        games_ml_df[c] = None

                st.dataframe(
                    games_ml_df[ml_cols].sort_values(
                        ["win_prob_pct", "avg_confidence"], ascending=[False, False]
                    ),
                    use_container_width=True,
                )

                st.markdown("#### Favorite per game (highest win_prob)")

                favorites = (
                    ml_df.sort_values("win_prob", ascending=False)
                    .groupby("game_key")
                    .head(1)
                    .sort_values("win_prob", ascending=False)
                )

                fav_ml_cols = [
                    "game_key",
                    "game_label",
                    "team",
                    "opponent",
                    "props_count",
                    "avg_confidence",
                    "avg_edge",
                    "win_prob_pct",
                    "ml_odds",
                    "model_spread",
                    "game_time",
                ]
                favorites["win_prob_pct"] = favorites["win_prob"] * 100.0
                favorites["ml_odds"] = favorites["win_prob"].apply(
                    lambda p: 1.0 / max(p, 0.0001)
                )

                # Display favorites + allow tracking ML bets
                if favorites.empty:
                    st.info("No favorite teams can be determined.")
                else:
                    favorites = favorites.copy()
                    for c in fav_ml_cols:
                        if c not in favorites.columns:
                            favorites[c] = None

                    st.markdown("##### Favorites table")

                    st.dataframe(
                        favorites[fav_ml_cols].sort_values(
                            "win_prob_pct", ascending=False
                        ),
                        use_container_width=True,
                    )

                    st.markdown("##### Game Cards (favorites)")

                    fav_list = favorites.to_dict(orient="records")

                    if not fav_list:
                        st.info("No games available for card view.")
                    else:
                        for idx, fav in enumerate(fav_list):
                            with st.container(border=True):
                                top_row1, top_row2 = st.columns([2, 1])
                                with top_row1:
                                    st.markdown(f"**{fav['game_label']}**")
                                    st.markdown(
                                        f"Favorite: **{fav['team']}** vs **{fav['opponent']}**"
                                    )
                                    st.markdown(
                                        f"Win %: **{fav['win_prob_pct']:.1f}%**  |  "
                                        f"Fair ML odds: **{fav['ml_odds']:.2f}**"
                                    )
                                    st.markdown(
                                        f"Model spread: **{fav['model_spread']:.1f}** points"
                                    )
                                with top_row2:
                                    if st.button(
                                        "Track favorite ML/spread",
                                        key=f"track_ml_{fav['game_key']}_{idx}",
                                    ):
                                        fake_row = pd.Series(
                                            {
                                                "player": None,
                                                "team": fav["team"],
                                                "opponent": fav["opponent"],
                                                "market": "ML/SPREAD",
                                                "line": fav["model_spread"],
                                                "book": "Model",
                                                "bet_side": "Over",  # as favorite
                                                "predicted_score": fav["model_spread"],
                                                "confidence_pct": min(
                                                    100.0,
                                                    max(
                                                        0.0,
                                                        fav["win_prob_pct"],
                                                    ),
                                                ),
                                                "ml_prob_over": fav["win_prob_pct"],
                                                "ml_odds_over": fav["ml_odds"],
                                                "game_time": fav["game_time"],
                                            }
                                        )
                                        track_bet(fake_row, ml=True, spread=True)

                    st.markdown("#### Game Moneyline Table (favorites only)")

                    ml_cols = [
                        "game_key",
                        "game_label",
                        "team",
                        "opponent",
                        "props_count",
                        "avg_confidence",
                        "avg_edge",
                        "win_prob_pct",
                        "ml_odds",
                    ]
                    for c in ml_cols:
                        if c not in favorites.columns:
                            favorites[c] = None
                    st.dataframe(
                        favorites[ml_cols].sort_values("win_prob_pct", ascending=False),
                        use_container_width=True,
                    )

                    st.markdown("#### Model Spreads (favorites per game)")
                    spread_cols = [
                        "game_key",
                        "game_label",
                        "team",
                        "opponent",
                        "props_count",
                        "avg_confidence",
                        "avg_edge",
                        "model_spread",
                        "game_time",
                    ]
                    for c in spread_cols:
                        if c not in favorites.columns:
                            favorites[c] = None
                    # Ensure model_spread is numeric before sorting by absolute value.
                    favorites["model_spread"] = pd.to_numeric(
                        favorites["model_spread"], errors="coerce"
                    )
                    st.dataframe(
                        favorites[spread_cols].sort_values(
                            "model_spread", key=lambda s: s.abs(), ascending=False
                        ),
                        use_container_width=True,
                    )


# =====================================================
# TAB 5: BET TRACKER (HIT/MISS + LEARNING) – same as v2.0.0
# =====================================================
with tab_bets:
    st.write("### Bet Tracker (results & learning from misses)")

    if st.button("Clear all tracked bets (reset)"):
        st.session_state["bet_tracker"] = []
        save_bets_to_disk([])
        st.success("All tracked bets cleared.")

    bets = st.session_state.get("bet_tracker", [])
    if not bets:
        st.info(
            "No bets tracked yet. On the Cards tab, click **'Track this bet'** "
            "or **'Track favorite ML/spread'** to add them."
        )
    else:
        bets_df = pd.DataFrame(bets).copy()

        # Allow marking results
        st.markdown("#### Mark bet results")
        for idx, row in bets_df.iterrows():
            cols = st.columns([4, 2, 1, 1, 1])
            with cols[0]:
                st.write(
                    f"{row.get('timestamp', '')[:19]} – "
                    f"{row.get('bet_type', '')} – "
                    f"{row.get('player') or row.get('team')} vs {row.get('opponent')} – "
                    f"{row.get('market')} @ {row.get('line')}"
                )
            with cols[1]:
                st.write(f"Side: {row.get('bet_side')}")
            with cols[2]:
                if st.button("Hit", key=f"hit_{idx}"):
                    bets_df.at[idx, "result"] = "Hit"
            with cols[3]:
                if st.button("Miss", key=f"miss_{idx}"):
                    bets_df.at[idx, "result"] = "Miss"
            with cols[4]:
                if st.button("Push", key=f"push_{idx}"):
                    bets_df.at[idx, "result"] = "Push"

        # Save updates if any result changed
        if not bets_df.equals(pd.DataFrame(bets)):
            st.session_state["bet_tracker"] = bets_df.to_dict(orient="records")
            save_bets_to_disk(st.session_state["bet_tracker"])
            st.success("Bet results updated.")

        st.markdown("#### Bet History")

        show_cols = [
            "timestamp",
            "bet_type",
            "player",
            "team",
            "opponent",
            "market",
            "line",
            "bet_side",
            "result",
            "confidence_pct",
            "ml_prob_over",
            "ml_odds_over",
            "game_time",
        ]
        for c in show_cols:
            if c not in bets_df.columns:
                bets_df[c] = None

        st.dataframe(
            bets_df[show_cols].sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

        # Simple learning: see how confidence relates to hit rate
        st.markdown("#### How is confidence performing?")

        completed = bets_df.dropna(subset=["result", "confidence_pct"])
        if completed.empty:
            st.info("No completed bets with confidence yet.")
        else:
            completed["conf_bucket"] = (
                (completed["confidence_pct"] // 10) * 10
            ).astype(int)
            perf = (
                completed.groupby("conf_bucket")["result"]
                .value_counts(normalize=True)
                .rename("rate")
                .reset_index()
                .pivot(index="conf_bucket", columns="result", values="rate")
                .fillna(0.0)
                .sort_index()
            )

            if mobile_mode:
                with st.expander("Hit/Miss rate by confidence bucket"):
                    st.dataframe(perf, use_container_width=True)
            else:
                st.markdown("**Hit/Miss rate by confidence bucket**")
                st.dataframe(perf, use_container_width=True)

            if "Hit" in perf.columns:
                best_bucket = perf["Hit"].idxmax()
                best_bucket_rate = perf["Hit"].max()
                st.caption(
                    f"Your strongest confidence zone so far is **{best_bucket}** "
                    f"({best_bucket_rate*100:.1f}% hits)."
                )
