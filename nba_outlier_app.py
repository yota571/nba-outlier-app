# =====================================================
# NBA Prop Edge Finder – PrizePicks
# Version 1.3.1
# - Presets, game filters, odds, trends, home/away splits
# - Bet Tracker tab (hit/miss + learning)
# - "Track this bet" button on cards
# - FIX: replace deprecated use_column_width with use_container_width
# =====================================================

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll

from DFS_Wrapper import PrizePick


# =====================================================
# STREAMLIT PAGE CONFIG & STATE
# =====================================================
st.set_page_config(
    page_title="NBA Outlier-Style App (PrizePicks)",
    layout="wide",
)

# simple in-memory bet tracker (can later be upgraded to CSV)
if "bet_tracker" not in st.session_state:
    st.session_state["bet_tracker"] = []


# =====================================================
# HELPERS: FILTER FULL-GAME PROPS ONLY
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
        "1h", " 1h ", "2h",
        "first half", "1st half", "second half", "2nd half",
        "half points", "half pts", "1h points", "1h pts",
        "1st quarter", "2nd quarter", "3rd quarter", "4th quarter",
        "first quarter", "second quarter", "third quarter", "fourth quarter",
        "q1", "q2", "q3", "q4",
        "first 5", "in first 5", "first five",
        "first 3 min", "first 6 min", "first 7 min",
        "in first six", "in first seven",
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
        pp = PrizePick()
        raw = pp.get_data(False)
    except Exception as e:
        st.error(f"Error loading PrizePicks data: {e}")
        return pd.DataFrame(
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
        )

    if not isinstance(raw, list):
        st.warning("PrizePicks data not in expected list format.")
        return pd.DataFrame(
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
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

        # map PrizePicks stat_type -> unified markets
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
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
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

    return df[["player_name", "team", "opponent", "market", "line", "game_time", "book"]]


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
# NBA STATS VIA nba_api
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

        # add simple home/away flag
        if "MATCHUP" in df.columns:
            df["IS_AWAY"] = df["MATCHUP"].astype(str).str.contains("@")
        else:
            df["IS_AWAY"] = False

        return df
    except Exception:
        return pd.DataFrame()


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
        return (
            gamelog_df["PTS"]
            + 1.2 * gamelog_df["REB"]
            + 1.5 * gamelog_df["AST"]
            + 3.0 * (gamelog_df["STL"] + gamelog_df["BLK"])
            - gamelog_df["TOV"]
        )

    return pd.Series(dtype="float")


def prob_to_american(p: float) -> str:
    """
    Convert probability (0–1) to American odds string.
    """
    try:
        p = float(p)
    except Exception:
        return "N/A"

    if p <= 0.0 or p >= 1.0:
        return "N/A"

    if p >= 0.5:
        odds = -round((p / (1 - p)) * 100)
    else:
        odds = round(((1 - p) / p) * 100)

    return f"{odds:+d}"


# =====================================================
# SIDEBAR UI
# =====================================================
st.sidebar.title("NBA Outlier-Style App (PrizePicks) – v1.3.1")

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

st.sidebar.markdown("### Quick filter preset")
preset = st.sidebar.selectbox(
    "Preset (optional)",
    ["Custom", "High Edge", "High Confidence", "Loose"],
    help=(
        "Custom: uses sliders\n"
        "High Edge: large edge, solid confidence\n"
        "High Confidence: high hit-rate, decent edge\n"
        "Loose: show more edges with looser thresholds"
    ),
)

# derive thresholds used in app based on preset
edge_threshold = min_edge
over_threshold = min_over_rate
conf_threshold = min_confidence

if preset == "High Edge":
    edge_threshold = max(edge_threshold, 3.0)
    over_threshold = max(over_threshold, 0.6)
    conf_threshold = max(conf_threshold, 65)
elif preset == "High Confidence":
    edge_threshold = max(edge_threshold, 1.5)
    over_threshold = max(over_threshold, 0.7)
    conf_threshold = max(conf_threshold, 70)
elif preset == "Loose":
    edge_threshold = min(edge_threshold, 0.5)
    over_threshold = min(over_threshold, 0.5)
    conf_threshold = min(conf_threshold, 50)

st.sidebar.markdown("---")
with st.sidebar.expander("What is Edge / Confidence / Odds / Bet side?"):
    st.write(
        "- **Edge** = how much higher the player has been performing vs the line (last N games).\n"
        "- **Confidence %** = blends hit-rate and edge size; higher = stronger data signal.\n"
        "- **Model odds** = fair American odds implied by the data (last N games).\n"
        "- **Bet side** = Over / Under / No clear edge, based on predicted score vs the line.\n"
        "- This app uses **full-game stats vs full-game PrizePicks props only.**"
    )

st.sidebar.markdown(
    "Markets used: **points, rebounds, assists, pra, ra (reb+ast), threes, fs (fantasy)**"
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

# create simple game label
props_df["game_label"] = props_df.apply(
    lambda r: f"{r['team']} vs {r['opponent']}"
    if pd.notna(r.get("team")) and pd.notna(r.get("opponent"))
    else "Unknown",
    axis=1,
)

if only_today and "game_time" in props_df.columns:
    today = datetime.today().date()
    props_df = props_df[props_df["game_time"].dt.date == today]

if props_df.empty:
    st.warning("No props after applying date filter.")
    st.stop()

# Top filters
teams = ["All teams"] + sorted(props_df["team"].dropna().unique().tolist())
markets = ["All markets"] + sorted(props_df["market"].dropna().unique().tolist())
games = ["All games"] + sorted(props_df["game_label"].dropna().unique().tolist())

top_col1, top_col2, top_col3, top_col4 = st.columns([2, 2, 3, 3])
with top_col1:
    team_filter = st.selectbox("Team filter", teams)
with top_col2:
    market_filter = st.selectbox("Market filter (points, pra, etc)", markets)
with top_col3:
    game_filter = st.selectbox("Game filter", games)
with top_col4:
    search_name = st.text_input("Search player (optional)", "")

df = props_df.copy()

if team_filter != "All teams":
    df = df[df["team"] == team_filter]
if market_filter != "All markets":
    df = df[df["market"].str.lower() == market_filter.lower()]
if game_filter != "All games":
    df = df[df["game_label"] == game_filter]

if df.empty:
    st.warning("No props match the selected filters.")
    st.stop()


# =====================================================
# EDGE / CONFIDENCE / PREDICTION / BET SIDE / ODDS
# =====================================================
st.title("NBA Prop Edge Finder (PrizePicks) – v1.3.1")
st.caption(
    "PrizePicks full-game props vs nba_api stats → edges, confidence %, prediction, model odds, trends, and bet tracking."
)

st.write("### Calculating edges…")

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
status_text.text("Computing edges, predictions, confidence, bet side…")

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

    # blended prediction
    w_season = 0.4
    w_last_n = 0.4
    w_last_7 = 0.2

    avg7_for_blend = avg_last_7 if avg_last_7 is not None else avg_last_n
    predicted_score = (
        w_season * season_avg
        + w_last_n * avg_last_n
        + w_last_7 * avg7_for_blend
    )

    hit_score = over_rate_n

    if line_float != 0:
        edge_ratio = max(0.0, edge_n / max(1.0, line_float))
    else:
        edge_ratio = 0.0
    edge_score = max(0.0, min(1.0, edge_ratio * 4.0))

    confidence = 0.6 * hit_score + 0.4 * edge_score
    confidence_pct = round(confidence * 100, 1)

    # Fair odds from over-rate
    over_prob = float(over_rate_n)
    over_prob_clamped = min(max(over_prob, 0.01), 0.99)
    under_prob = 1.0 - over_prob_clamped

    over_odds = prob_to_american(over_prob_clamped)
    under_odds = prob_to_american(under_prob)

    delta = predicted_score - line_float
    if delta >= 0.5:
        bet_side = "Over"
    elif delta <= -0.5:
        bet_side = "Under"
    else:
        bet_side = "No clear edge"

    if bet_side == "Over":
        bet_odds = over_odds
    elif bet_side == "Under":
        bet_odds = under_odds
    else:
        bet_odds = "N/A"

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
            "over_prob": round(over_prob_clamped, 3),
            "under_prob": round(under_prob, 3),
            "over_odds": over_odds,
            "under_odds": under_odds,
            "bet_odds": bet_odds,
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

edge_cols = [c for c in edges_df.columns if c.startswith("edge_last_") and "7" not in c]
rate_cols = [c for c in edges_df.columns if c.startswith("over_rate_last_") and "7" not in c]

if not edge_cols or not rate_cols:
    st.error("Edge/over_rate columns missing from results.")
    st.stop()

edge_col = edge_cols[0]
rate_col = rate_cols[0]

# Apply player search filter (for card + table views)
if search_name.strip():
    view_df = edges_df[edges_df["player"].str.contains(search_name, case=False, na=False)]
else:
    view_df = edges_df.copy()

if view_df.empty:
    st.warning("No props after applying search filter.")
    st.stop()


# =====================================================
# TABS: CARDS + EXPLANATION / TABLE / PLAYER DETAIL / BETS
# =====================================================
tab_cards, tab_table, tab_player, tab_bets = st.tabs(
    ["Cards & Explanation", "Table", "Player Detail", "Bet Tracker"]
)


# =====================================================
# TAB 1: CARD VIEW (WITH HEADSHOTS) + “WHY THIS PROP?” + TRACK BET
# =====================================================
with tab_cards:
    st.write("### Featured Edges (Card View)")

    filtered_edges = view_df[
        (view_df[rate_col] >= over_threshold)
        & (view_df[edge_col] >= edge_threshold)
        & (view_df["confidence_pct"] >= conf_threshold)
    ]

    if filtered_edges.empty:
        featured_df = view_df.copy()
        st.caption(
            "No props match all filters/preset yet – showing best available edges instead."
        )
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
                                    # FIX: use_container_width instead of deprecated use_column_width
                                    st.image(img_url, use_container_width=True)
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

                            st.markdown(
                                f"Predicted: **{r['predicted_score']}**  "
                                f"(Season avg: {r['season_avg']})"
                            )

                        # bottom row: metrics + confidence bar + odds
                        conf = r.get("confidence_pct", 0) or 0
                        edge_val = r.get(edge_col, 0) or 0
                        hit = r.get(rate_col, 0) or 0

                        st.markdown(
                            f"Edge: `{edge_val:.2f}` &nbsp; | &nbsp; "
                            f"Hit rate (N): `{hit:.2f}` &nbsp; | &nbsp; "
                            f"Confidence: `{conf:.1f}%`"
                        )
                        st.progress(min(max(conf / 100.0, 0.0), 1.0))

                        side = r.get("bet_side", "No clear edge")
                        if side == "Over":
                            side_prob = r.get("over_prob", 0)
                            side_odds = r.get("over_odds", "N/A")
                        elif side == "Under":
                            side_prob = r.get("under_prob", 0)
                            side_odds = r.get("under_odds", "N/A")
                        else:
                            side_prob = None
                            side_odds = "N/A"

                        if side_prob is not None:
                            st.markdown(
                                f"Model {side} odds: `{side_odds}` "
                                f"({side_prob*100:.1f}% implied)"
                            )

                        # Explanation button
                        if st.button(
                            "Why this prop?",
                            key=f"why_{r['player']}_{r['market']}_{k}",
                        ):
                            st.session_state["explain_row"] = r.to_dict()

                        # --- Track this bet button ---
                        if st.button(
                            "Track this bet",
                            key=f"track_{r['player']}_{r['market']}_{k}",
                        ):
                            bet = {
                                "player": r["player"],
                                "team": r.get("team"),
                                "opponent": r.get("opponent"),
                                "market": r["market"],
                                "line": r["line"],
                                "bet_side": r.get("bet_side"),
                                "bet_odds": r.get("bet_odds"),
                                "predicted_score": r.get("predicted_score"),
                                "confidence_pct": r.get("confidence_pct"),
                                "game_time": r.get("game_time"),
                            }
                            st.session_state["bet_tracker"].append(bet)
                            st.success("Bet added to tracker for this game.")

    st.write("### Prop Explanation (Why this prediction, confidence & odds)")

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
            over_odds = r.get("over_odds")
            under_odds = r.get("under_odds")
            bet_odds = r.get("bet_odds")

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Key Numbers**")
                st.write(f"- Season average: **{season_avg}**")
                st.write(f"- Last {games_to_look_back} avg: **{avg_last_n}**")
                st.write(f"- Hit rate last {games_to_look_back}: **{over_rate_n:.2f}**")
                st.write(f"- Edge last {games_to_look_back}: **{edge_n:.2f}** vs line **{line}**")
                if avg_last_7 is not None:
                    st.write(f"- Last 7 avg: **{avg_last_7}**")
                    st.write(f"- Hit rate last 7: **{(over_rate_7 or 0):.2f}**")
                    st.write(f"- Edge last 7: **{(edge_7 or 0):.2f}**")

                # smarter context: home/away splits and vs-opponent
                if player_name in player_logs:
                    gamelog = player_logs[player_name]
                    full_series = get_market_series(gamelog, market).dropna()

                    if not gamelog.empty and not full_series.empty:
                        home_mask = gamelog["IS_AWAY"] == False
                        away_mask = gamelog["IS_AWAY"] == True
                        if home_mask.any():
                            home_avg = full_series[home_mask].mean()
                            st.write(f"- Home avg: **{home_avg:.2f}**")
                        if away_mask.any():
                            away_avg = full_series[away_mask].mean()
                            st.write(f"- Away avg: **{away_avg:.2f}**")

                        opp_today = r.get("opponent")
                        if opp_today and "MATCHUP" in gamelog.columns:
                            opp_mask = gamelog["MATCHUP"].astype(str).str.contains(
                                str(opp_today), case=False, na=False
                            )
                            if opp_mask.any():
                                opp_avg = full_series[opp_mask].mean()
                                st.write(
                                    f"- Vs {opp_today} avg (sample): **{opp_avg:.2f}**"
                                )

            with col_right:
                st.markdown("**Model Result**")
                st.write(f"- Predicted score: **{predicted_score}**")
                st.write(f"- Confidence: **{confidence_pct:.1f}%**")
                st.write(f"- Model Over odds: **{over_odds}**")
                st.write(f"- Model Under odds: **{under_odds}**")
                st.write(f"- Recommended side: **{bet_side}**  (odds: **{bet_odds}**)")

                st.markdown("**How confidence & odds are calculated:**")
                st.write(
                    "- Hit rate over the last N games (how often they went over the line).\n"
                    "- Edge: how far the average is **above** the line.\n"
                    "- Confidence = 0.6 × HitRate + 0.4 × EdgeStrength.\n"
                    "- Model odds: convert the hit-rate to fair American odds."
                )

            # Game-by-game history + trend chart
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
                        "GAME_DATE": gamelog["GAME_DATE"].iloc[:games_to_look_back].values,
                        market.upper(): last_n_vals.values,
                        "vs_line": [
                            "Over" if v > line else "Under" if v < line else "Push"
                            for v in last_n_vals.values
                        ],
                    }
                ).sort_values("GAME_DATE", ascending=True)

                st.markdown(f"**Last {games_to_look_back} games for {market.upper()}**")
                st.dataframe(hist_df, use_container_width=True)

                st.markdown("**Trend over last N games**")
                trend_df = hist_df.set_index("GAME_DATE")[[market.upper()]]
                st.line_chart(trend_df)
            else:
                st.info("Not enough recent games to show game-by-game history.")
    else:
        st.caption("Click **“Why this prop?”** on any card above to see a detailed explanation here.")


# =====================================================
# TAB 2: FULL EDGES TABLE (with player filter)
# =====================================================
with tab_table:
    st.write("### Full Edges Table")

    # Player filter on the table tab
    table_player_options = ["All players"] + sorted(
        view_df["player"].dropna().unique().tolist()
    )
    selected_table_player = st.selectbox(
        "Filter table by player (optional)",
        table_player_options,
        key="table_player_filter",
    )

    def highlight_edges(row, e_thr=edge_threshold, o_thr=over_threshold, c_thr=conf_threshold):
        styles = [""] * len(row)
        try:
            val = float(row[edge_col])
            over = float(row[rate_col])
            conf = float(row["confidence_pct"])
            side = str(row["bet_side"])
        except Exception:
            return styles

        if val >= e_thr and over >= o_thr and conf >= c_thr and side in ("Over", "Under"):
            color = "background-color: #d4f8d4"   # green for strong edges
        elif val < 0:
            color = "background-color: #f8d4d4"   # red if underperforming line
        elif val > 0:
            color = "background-color: #fff4c2"   # yellow if mild positive
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
        "bet_odds",
        "predicted_score",
        "season_avg",
        f"avg_last_{games_to_look_back}",
        f"over_rate_last_{games_to_look_back}",
        f"edge_last_{games_to_look_back}",
        "avg_last_7",
        "over_rate_last_7",
        "edge_last_7",
        "over_prob",
        "over_odds",
        "under_prob",
        "under_odds",
        "confidence_pct",
        "game_time",
    ]

    # base table is the search-filtered view_df
    table_df = view_df.copy()
    for col in display_cols:
        if col not in table_df.columns:
            table_df[col] = None

    # apply player filter (if not "All players")
    if selected_table_player != "All players":
        table_df_filtered = table_df[table_df["player"] == selected_table_player].copy()
    else:
        table_df_filtered = table_df.copy()

    if table_df_filtered.empty:
        st.warning("No rows for this player with the current filters.")
    else:
        styled_edges = (
            table_df_filtered[display_cols]
            .sort_values(by=["confidence_pct", rate_col, edge_col], ascending=False)
            .style.apply(highlight_edges, axis=1)
        )

        st.dataframe(styled_edges, use_container_width=True)

    # Show a compact "all props for this player" table under the main grid
    if selected_table_player != "All players":
        st.markdown(f"#### All props for **{selected_table_player}**")
        player_all_lines = edges_df[edges_df["player"] == selected_table_player].copy()
        if not player_all_lines.empty:
            player_cols = [
                "market",
                "line",
                "book",
                "bet_side",
                "bet_odds",
                "predicted_score",
                "confidence_pct",
                f"over_rate_last_{games_to_look_back}",
                f"edge_last_{games_to_look_back}",
                "game_time",
            ]
            for c in player_cols:
                if c not in player_all_lines.columns:
                    player_all_lines[c] = None

            st.dataframe(
                player_all_lines[player_cols]
                .sort_values(
                    by=["confidence_pct", f"over_rate_last_{games_to_look_back}"],
                    ascending=False,
                ),
                use_container_width=True,
            )
        else:
            st.info("No props found for this player in the full edges set.")

    if errors:
        with st.expander("Warnings/errors (players/markets skipped)"):
            st.caption(
                "These players either aren't in nba_api yet or have no game log data. "
                "They are skipped but do not break the app."
            )
            for e in errors:
                st.write("- ", e)


# =====================================================
# TAB 3: PLAYER DETAIL (RAW GAME LOG + TREND CHART)
# =====================================================
with tab_player:
    st.write("### Player Detail (Raw Game Log & Trends)")

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
            st.dataframe(gamelog.head(games_to_look_back), use_container_width=True)

            st.markdown("#### Trend by market")
            market_choice = st.selectbox(
                "Market to chart",
                ["points", "rebounds", "assists", "pra", "ra", "threes", "fs"],
                index=0,
                key="player_detail_market",
            )

            series = get_market_series(gamelog, market_choice).dropna()
            if not series.empty:
                last_n = series.iloc[:games_to_look_back]
                chart_df = pd.DataFrame(
                    {
                        "GAME_DATE": gamelog["GAME_DATE"].iloc[:len(last_n)].values,
                        market_choice.upper(): last_n.values,
                    }
                ).sort_values("GAME_DATE", ascending=True)
                st.line_chart(chart_df.set_index("GAME_DATE"))
            else:
                st.info("No data for this market for this player.")
        else:
            st.warning("No game log available for this player.")


# =====================================================
# TAB 4: BET TRACKER (HIT/MISS + LEARNING)
# =====================================================
with tab_bets:
    st.write("### Bet Tracker (results & learning from misses)")

    bets = st.session_state.get("bet_tracker", [])
    if not bets:
        st.info(
            "No bets tracked yet. On the Cards tab, click **'Track this bet'** "
            "to add one after you decide to play it."
        )
    else:
        bets_df = pd.DataFrame(bets).copy()

        # Ensure game_time is datetime
        if "game_time" in bets_df.columns:
            bets_df["game_time"] = pd.to_datetime(bets_df["game_time"], errors="coerce")

        actual_vals = []
        results = []

        for _, b in bets_df.iterrows():
            player_name = b.get("player")
            market = b.get("market")
            line = b.get("line")
            side = b.get("bet_side")
            gtime = b.get("game_time")

            actual_stat = None
            result = "Unknown"

            try:
                if pd.isna(gtime):
                    result = "No game_time"
                else:
                    game_date = gtime.date()

                    # fetch gamelog
                    pid = get_player_id(player_name)
                    glog = get_player_gamelog(pid)
                    if glog.empty:
                        result = "No game log"
                    else:
                        glog = glog.copy()
                        glog["GAME_DATE_ONLY"] = glog["GAME_DATE"].dt.date
                        game_row = glog[glog["GAME_DATE_ONLY"] == game_date]

                        if game_row.empty:
                            result = "Not played / not final yet"
                        else:
                            # compute actual stat from that single game row
                            series_one = get_market_series(game_row, market).dropna()
                            if series_one.empty:
                                result = "No stat for market"
                            else:
                                actual_stat = float(series_one.iloc[0])

                                if side == "Over":
                                    result = "Hit" if actual_stat > line else "Miss"
                                elif side == "Under":
                                    result = "Hit" if actual_stat < line else "Miss"
                                else:
                                    result = "N/A"
            except Exception:
                if result == "Unknown":
                    result = "Error"

            actual_vals.append(actual_stat)
            results.append(result)

        bets_df["actual_stat"] = actual_vals
        bets_df["result"] = results

        st.markdown("#### Bet results")
        show_cols = [
            "player",
            "team",
            "opponent",
            "market",
            "line",
            "bet_side",
            "bet_odds",
            "predicted_score",
            "actual_stat",
            "confidence_pct",
            "game_time",
            "result",
        ]
        for c in show_cols:
            if c not in bets_df.columns:
                bets_df[c] = None

        st.dataframe(bets_df[show_cols], use_container_width=True)

        # ---------- LEARNING FROM MISSES ----------
        st.markdown("### Learning from misses")

        hits_df = bets_df[bets_df["result"].isin(["Hit", "Miss"])].copy()
        if hits_df.empty:
            st.info("No completed bets yet to learn from.")
        else:
            # Hit rate by market
            market_perf = (
                hits_df.groupby("market")["result"]
                .value_counts(normalize=True)
                .rename("rate")
                .reset_index()
                .pivot(index="market", columns="result", values="rate")
                .fillna(0.0)
            )
            st.markdown("**Hit/Miss rate by market**")
            st.dataframe(market_perf, use_container_width=True)

            # Hit rate by confidence bucket
            def conf_bucket(x):
                try:
                    c = float(x)
                except Exception:
                    return "Unknown"
                if c < 55:
                    return "<55%"
                elif c < 60:
                    return "55–60%"
                elif c < 65:
                    return "60–65%"
                elif c < 70:
                    return "65–70%"
                elif c < 80:
                    return "70–80%"
                else:
                    return "80%+"

            hits_df["conf_bucket"] = hits_df["confidence_pct"].apply(conf_bucket)
            bucket_perf = (
                hits_df.groupby("conf_bucket")["result"]
                .value_counts(normalize=True)
                .rename("rate")
                .reset_index()
                .pivot(index="conf_bucket", columns="result", values="rate")
                .fillna(0.0)
                .sort_index()
            )
            st.markdown("**Hit/Miss rate by confidence bucket**")
            st.dataframe(bucket_perf, use_container_width=True)
