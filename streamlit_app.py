import time
from datetime import datetime

import pandas as pd
import streamlit as st

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll

from DFS_Wrapper import PrizePick


# =====================================================
# STREAMLIT PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="NBA Outlier-Style App (PrizePicks, Cloud Version)",
    layout="wide",
)


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
# PRIZEPICKS RAW LOADER (SHARED BY EDGE + ALT-LINE VIEWS)
# =====================================================
@st.cache_data
def get_prizepicks_raw() -> list[dict]:
    """
    Get raw PrizePicks data via DFS_Wrapper.PrizePick.

    This is the single source; other loaders (full-game props, alt-line browser)
    use this so everything stays in sync.
    """
    try:
        pp = PrizePick()
        raw = pp.get_data(organize_data=False)
    except TypeError:
        pp = PrizePick()
        raw = pp.get_data(False)
    except Exception as e:
        st.error(f"Error loading PrizePicks data: {e}")
        return []

    if not isinstance(raw, list):
        st.warning("PrizePicks data not in expected list format.")
        return []

    return raw


# =====================================================
# PRIZEPICKS LOADER (NBA, FULL GAME ONLY) - USED BY EDGE FINDER
# =====================================================
@st.cache_data
def load_prizepicks_nba_props() -> pd.DataFrame:
    """
    Pull cleaned NBA props from PrizePicks for the edge finder.

    - NBA only
    - pre-game only
    - full-game only (no halves/quarters/first 5 min)
    - keep ONLY one line per (player, team, opponent, market) – highest line
    """
    raw = get_prizepicks_raw()
    if not raw:
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

    df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    df = df.groupby(
        ["player_name", "team", "opponent", "market"],
        as_index=False,
    ).last()

    return df[["player_name", "team", "opponent", "market", "line", "game_time", "book"]]


# =====================================================
# ALT-LINE LOADER (NBA, FULL GAME ONLY, ALL LINES)
# =====================================================
@st.cache_data
def load_prizepicks_alt_lines() -> pd.DataFrame:
    """
    Build a DF of *all* full-game lines per (player, market), including alt lines.

    Used only for the Alt Line Browser tab. Shares the same filters as the main props:
    - NBA only
    - pre-game only
    - full-game only
    but does NOT collapse to one line per combo.
    """
    raw = get_prizepicks_raw()
    if not raw:
        return pd.DataFrame(
            columns=[
                "player_name",
                "team",
                "opponent",
                "market",
                "stat_type_raw",
                "line",
                "odds_type",
                "game_time",
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
        odds_type = item.get("odds_type")

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

        if "+" in player_name:
            continue
        if any(sep in opponent for sep in ("/", "|", "+")):
            continue

        records.append(
            {
                "player_name": player_name,
                "team": team,
                "opponent": opponent,
                "market": market,
                "stat_type_raw": stat_type,
                "line": line_val,
                "odds_type": odds_type,
                "game_time": start_time,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "player_name",
                "team",
                "opponent",
                "market",
                "stat_type_raw",
                "line",
                "odds_type",
                "game_time",
            ]
        )

    df = pd.DataFrame(records)
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df.dropna(subset=["line"])
    df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df


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


# =====================================================
# SIDEBAR UI
# =====================================================
st.sidebar.title("NBA Outlier-Style App (PrizePicks, Cloud)")

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


# =====================================================
# LOAD PROPS
# =====================================================
props_df = None
alt_df = None  # for alt-line browser

if mode == "PrizePicks (live)":
    props_df = load_prizepicks_nba_props()
    alt_df = load_prizepicks_alt_lines()
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
    props_df = props_df[props_df["game_time"].dt.date == today]

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
# EDGE / CONFIDENCE / PREDICTION / BET SIDE
# =====================================================
st.title("NBA Prop Edge Finder (PrizePicks, Cloud Version)")
st.caption("PrizePicks full-game props vs nba_api stats → edges, confidence %, & prediction.")

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
# TABS: CARDS + EXPLANATION / TABLE / PLAYER DETAIL / ALT-LINE BROWSER
# =====================================================
tab_cards, tab_table, tab_player, tab_alt = st.tabs(
    ["Cards & Explanation", "Table", "Player Detail", "Alt Line Browser"]
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

            with col_right:
                st.markdown("**Model Result**")
                st.write(f"- Predicted score: **{predicted_score}**")
                st.write(f"- Confidence: **{confidence_pct:.1f}%**")
                st.write(f"- Recommended side: **{bet_side}**")

                st.markdown("**How confidence is calculated:**")
                st.write(
                    "- Take the hit rate over the last N games (how often they went over the line).\n"
                    "- Measure how far the average is **above** the line (edge).\n"
                    "- Combine them: `Confidence = 0.6 * HitRate + 0.4 * EdgeStrength`, then shown as a %."
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
                        "GAME_DATE": gamelog["GAME_DATE"].iloc[:games_to_look_back].values,
                        market.upper(): last_n_vals.values,
                        "vs_line": [
                            "Over" if v > line else "Under" if v < line else "Push"
                            for v in last_n_vals.values
                        ],
                    }
                )
                st.markdown(f"**Last {games_to_look_back} games for {market.upper()}**")
                st.dataframe(hist_df, use_container_width=True)
            else:
                st.info("Not enough recent games to show game-by-game history.")
    else:
        st.caption("Click **“Why this prop?”** on any card above to see a detailed explanation here.")


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

        if val >= min_edge and over >= min_over_rate and conf >= min_confidence and side in ("Over", "Under"):
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
        "predicted_score",
        "season_avg",
        f"avg_last_{games_to_look_back}",
        f"over_rate_last_{games_to_look_back}",
        f"edge_last_{games_to_look_back}",
        "avg_last_7",
        "over_rate_last_7",
        "edge_last_7",
        "confidence_pct",
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
            st.dataframe(gamelog.head(games_to_look_back), use_container_width=True)
        else:
            st.warning("No game log available for this player.")


# =====================================================
# TAB 4: ALT LINE BROWSER (PRIZEPICKS-STYLE)
# =====================================================
with tab_alt:
    st.write("### Alt Line Browser (PrizePicks-style)")

    if mode != "PrizePicks (live)":
        st.info("Alt line browser is only available in **PrizePicks (live)** mode.")
    else:
        if alt_df is None or alt_df.empty:
            st.warning("No alt-line data available.")
        else:
            # Apply same top filters & search to alt_df
            alt_view = alt_df.copy()

            if team_filter != "All":
                alt_view = alt_view[alt_view["team"] == team_filter]
            if market_filter != "All":
                alt_view = alt_view[
                    alt_view["market"].str.lower() == market_filter.lower()
                ]
            if search_name.strip():
                alt_view = alt_view[
                    alt_view["player_name"].str.contains(search_name, case=False, na=False)
                ]

            if alt_view.empty:
                st.warning("No alt lines match the current filters/search.")
            else:
                # Cards per (player_name, market)
                cards_df = (
                    alt_view.groupby(["player_name", "team", "market"], as_index=False)
                    .agg(
                        lines_count=("line", "count"),
                        min_line=("line", "min"),
                        max_line=("line", "max"),
                    )
                    .sort_values("player_name")
                )

                if "alt_selected_player" not in st.session_state:
                    st.session_state["alt_selected_player"] = None
                if "alt_selected_market" not in st.session_state:
                    st.session_state["alt_selected_market"] = None

                CARDS_PER_ROW = 3
                st.markdown("#### Player Props (click to view all alt lines)")

                for i in range(0, len(cards_df), CARDS_PER_ROW):
                    row_slice = cards_df.iloc[i : i + CARDS_PER_ROW]
                    cols = st.columns(len(row_slice))

                    for idx, (_, row_c) in enumerate(row_slice.iterrows()):
                        with cols[idx]:
                            pn = row_c["player_name"]
                            tm = row_c["team"]
                            mk = row_c["market"]
                            lc = int(row_c["lines_count"])
                            mn = row_c["min_line"]
                            mx = row_c["max_line"]

                            # headshot if we have player_id from earlier
                            pid = player_ids.get(pn)
                            if pid:
                                try:
                                    img_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{int(pid)}.png"
                                    st.image(img_url, use_column_width=True)
                                except Exception:
                                    st.write(" ")
                            else:
                                st.write(" ")

                            st.markdown(f"**{pn}**")
                            st.markdown(f"🧢 Team: `{tm}`")
                            st.markdown(f"📊 Market: **{mk}**")
                            st.markdown(f"🔢 Alt lines: `{lc}`")
                            if pd.notna(mn) and pd.notna(mx):
                                st.markdown(f"Range: `{mn}` — `{mx}`")

                            if st.button(
                                "View all lines",
                                key=f"alt_{pn}_{mk}_{i}_{idx}",
                            ):
                                st.session_state["alt_selected_player"] = pn
                                st.session_state["alt_selected_market"] = mk

                st.markdown("---")
                st.write("#### Selected Prop Alt Lines")

                sel_player = st.session_state.get("alt_selected_player")
                sel_market = st.session_state.get("alt_selected_market")

                if sel_player and sel_market:
                    st.markdown(
                        f"Showing all alt lines for **{sel_player} — {sel_market}**"
                    )
                    detail_alt = alt_view[
                        (alt_view["player_name"] == sel_player)
                        & (alt_view["market"] == sel_market)
                    ].copy()

                    if detail_alt.empty:
                        st.warning("No lines found (filters may have changed).")
                    else:
                        detail_alt = detail_alt.sort_values("line")
                        show_cols = [
                            "line",
                            "odds_type",
                            "stat_type_raw",
                            "game_time",
                            "opponent",
                        ]
                        show_cols = [
                            c for c in show_cols if c in detail_alt.columns
                        ]
                        st.dataframe(
                            detail_alt[show_cols], use_container_width=True
                        )
                else:
                    st.caption(
                        "Click **“View all lines”** on any player card above "
                        "to see every alt line for that prop."
                    )
