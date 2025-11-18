import time
from datetime import datetime

import pandas as pd
import streamlit as st

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll

from DFS_Wrapper import PrizePick


# -------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="NBA Outlier-Style App",
    layout="wide",
)


# -------------------------------------------------
# HELPERS: PROPS LOADING
# -------------------------------------------------
@st.cache_data
def load_props_from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    required_cols = ["player_name", "team", "opponent", "market", "line", "game_time"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    if "game_time" in df.columns:
        df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df[required_cols]


@st.cache_data
def load_prizepicks_nba_props() -> pd.DataFrame:
    """Pull cleaned NBA props from PrizePicks."""
    try:
        pp = PrizePick()
        raw = pp.get_data(organize_data=False)
    except TypeError:
        pp = PrizePick()
        raw = pp.get_data(False)
    except Exception as e:
        st.error(f"Error loading PrizePicks data: {e}")
        return pd.DataFrame(columns=["player_name", "team", "opponent", "market", "line", "game_time"])

    if not isinstance(raw, list):
        st.warning("PrizePicks data not in expected list format.")
        return pd.DataFrame(columns=["player_name", "team", "opponent", "market", "line", "game_time"])

    records = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        league = item.get("league", "")
        if "NBA" not in str(league).upper():
            continue

        # pre-game only
        status = str(item.get("status", "")).lower()
        if status and status not in ("pre_game", "pre-game", "pregame"):
            continue

        stat_type = item.get("stat_type")

        # line value
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

        # start time / game time
        start_time = item.get("start_time") or item.get("game_date_time")

        # Map PrizePicks stat_type -> our markets
        market_map = {
            "Points": "points",
            "Rebounds": "rebounds",
            "Assists": "assists",
            "Pts + Rebs + Asts": "pra",
            "Pts+Rebs+Asts": "pra",
            "Rebs+Asts": "ra",
            "Rebs + Asts": "ra",
            "3-Pointers Made": "threes",
            "Fantasy Score": "fs",  # fantasy score
        }
        market = market_map.get(stat_type)
        if market is None:
            # Skip stuff we don't handle
            continue

        # Skip multi-player combos like "Maxey + Harden"
        if "+" in player_name:
            continue

        # Skip weird opponent strings (multi-team / cross-game promos)
        if any(sep in opponent for sep in ("/", "|", "+")):
            continue

        # Filter out obviously tiny alt-lines/promos (keep main full-game)
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
        # threes often small (1.5, 2.5) → keep

        records.append({
            "player_name": player_name,
            "team": team,
            "opponent": opponent,
            "market": market,
            "line": line_val,
            "game_time": start_time,
        })

    if not records:
        return pd.DataFrame(columns=["player_name", "team", "opponent", "market", "line", "game_time"])

    df = pd.DataFrame(records)

    # Dedupe: keep highest line for each player/team/opponent/market combo
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df.dropna(subset=["line"])

    df = df.sort_values("line")  # so groupby().last() keeps highest line
    df = df.groupby(
        ["player_name", "team", "opponent", "market"],
        as_index=False
    ).last()

    # Parse game_time to datetime
    df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    # Ensure all columns exist and in order
    for col in ["player_name", "team", "opponent", "market", "line", "game_time"]:
        if col not in df.columns:
            df[col] = None

    return df[["player_name", "team", "opponent", "market", "line", "game_time"]]


# -------------------------------------------------
# HELPERS: NBA STATS VIA nba_api
# -------------------------------------------------
@st.cache_data
def get_all_players():
    return players.get_players()


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation and extra spaces for loose matching."""
    if not name:
        return ""
    name = name.lower()
    for ch in [".", ",", "'", "`"]:
        name = name.replace(ch, "")
    name = " ".join(name.split())
    return name


@st.cache_data
def get_player_id(player_name: str):
    """Get NBA player ID with loose name matching."""
    all_players = get_all_players()
    if not player_name:
        return None

    target = normalize_name(player_name)

    # Exact normalized match
    for p in all_players:
        if normalize_name(p["full_name"]) == target:
            return p["id"]

    # Contains match (handles minor differences)
    for p in all_players:
        norm = normalize_name(p["full_name"])
        if target and (target in norm or norm in target):
            return p["id"]

    return None


@st.cache_data
def get_player_gamelog(player_id: int) -> pd.DataFrame:
    """Get full player game log (all seasons), most recent first."""
    if player_id is None:
        return pd.DataFrame()
    try:
        gl = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=SeasonAll.all
        )
        df = gl.get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        df = df.sort_values("GAME_DATE", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


def get_market_series(gamelog_df: pd.DataFrame, market: str) -> pd.Series:
    """
    Return per-game values for the requested market.
    Fantasy score (fs) uses PrizePicks NBA scoring:
    FS = PTS + 1.2*REB + 1.5*AST + 3*(STL+BLK) - TOV
    """
    market = (market or "").lower().strip()
    if gamelog_df.empty:
        return pd.Series(dtype="float")

    # Ensure needed columns exist
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
    if market == "ra":  # Rebounds + Assists
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


# -------------------------------------------------
# SIDEBAR UI
# -------------------------------------------------
st.sidebar.title("NBA Outlier-Style App")

mode = st.sidebar.radio(
    "Where should props come from?",
    ["PrizePicks (live)", "Upload CSV manually"],
)

games_to_look_back = st.sidebar.slider(
    "Games to look back (N)",
    min_value=5,
    max_value=25,
    value=10,
    step=1,
)

min_over_rate = st.sidebar.slider(
    "Minimum Over % (last N games)",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
)

min_edge = st.sidebar.number_input(
    "Minimum Edge (Avg - Line)",
    value=1.0,
    step=0.5,
)

# Default OFF so you always see props; turn it on to hide tomorrow/future slates
only_today = st.sidebar.checkbox(
    "Only today's games (by game_time)",
    value=False,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Supported markets: **points, rebounds, assists, pra, ra (reb+ast), threes, fs (fantasy score)**"
)


# -------------------------------------------------
# LOAD PROPS
# -------------------------------------------------
props_df = None

if mode == "PrizePicks (live)":
    st.sidebar.success("Using live NBA props from PrizePicks.")
    props_df = load_prizepicks_nba_props()
else:
    uploaded = st.sidebar.file_uploader(
        "Upload props CSV",
        type=["csv"],
        help="Must contain columns: player_name, team, opponent, market, line, game_time",
    )
    if uploaded:
        props_df = load_props_from_csv(uploaded)

if props_df is None or props_df.empty:
    st.info("No props loaded yet. Choose PrizePicks mode or upload a CSV.")
    st.stop()

if only_today and "game_time" in props_df.columns:
    today = datetime.today().date()
    props_df = props_df[props_df["game_time"].dt.date == today]

if props_df.empty:
    st.warning("No props after applying date filter.")
    st.stop()


# -------------------------------------------------
# TOP FILTERS
# -------------------------------------------------
teams = ["All"] + sorted(props_df["team"].dropna().unique().tolist())
markets = ["All"] + sorted(props_df["market"].dropna().unique().tolist())

col1, col2 = st.columns(2)
with col1:
    team_filter = st.selectbox("Team filter", teams)
with col2:
    market_filter = st.selectbox("Market filter", markets)

df = props_df.copy()

if team_filter != "All":
    df = df[df["team"] == team_filter]

if market_filter != "All":
    df = df[df["market"].str.lower() == market_filter.lower()]

if df.empty:
    st.warning("No props match the selected filters.")
    st.stop()


# -------------------------------------------------
# EDGE CALCULATION
# -------------------------------------------------
st.title("NBA Prop Edge Finder")
st.caption("Pulls PrizePicks NBA props (or CSV) and uses nba_api stats to compute simple edges.")

st.write("### Calculating edges…")

rows = []
errors = []

# Build cache of gamelog per unique player
unique_players = sorted(df["player_name"].dropna().unique().tolist())
player_logs = {}

progress = st.progress(0.0)
status_text = st.empty()

total_players = len(unique_players) if unique_players else 1

for i, name in enumerate(unique_players):
    status_text.text(f"Fetching NBA stats for players… ({i+1}/{total_players})")

    pid = get_player_id(name)
    if pid is None:
        errors.append(f"Player not found in nba_api: {name}")
        continue

    glog = get_player_gamelog(pid)
    if glog.empty:
        errors.append(f"No game log for player: {name}")
        continue

    player_logs[name] = glog

    time.sleep(0.2)  # be gentle to API

    progress.progress((i + 1) / total_players)

progress.progress(1.0)
status_text.text("Computing edges for each prop…")

for _, row in df.iterrows():
    player_name = row.get("player_name")
    market = row.get("market")
    line = row.get("line")

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

    # Season average
    season_avg = series.mean()

    # Last N (slider)
    last_n = series.iloc[:games_to_look_back]
    if last_n.empty:
        errors.append(f"No recent games for {player_name}")
        continue

    avg_last_n = last_n.mean()
    over_rate_n = (last_n > line_float).mean()
    edge_n = avg_last_n - line_float

    # Last 7 fixed window
    last7 = series.iloc[:7]
    if len(last7) > 0:
        avg_last_7 = last7.mean()
        over_rate_7 = (last7 > line_float).mean()
        edge_7 = avg_last_7 - line_float
    else:
        avg_last_7 = None
        over_rate_7 = None
        edge_7 = None

    rows.append({
        "player": player_name,
        "team": row.get("team"),
        "opponent": row.get("opponent"),
        "market": market,
        "line": line_float,
        "season_avg": round(season_avg, 2),
        f"avg_last_{games_to_look_back}": round(avg_last_n, 2),
        f"over_rate_last_{games_to_look_back}": round(over_rate_n, 2),
        f"edge_last_{games_to_look_back}": round(edge_n, 2),
        "avg_last_7": round(avg_last_7, 2) if avg_last_7 is not None else None,
        "over_rate_last_7": round(over_rate_7, 2) if over_rate_7 is not None else None,
        "edge_last_7": round(edge_7, 2) if edge_7 is not None else None,
        "game_time": row.get("game_time"),
    })

if not rows:
    st.error("No edges could be calculated from the current props.")
    if errors:
        with st.expander("Show warnings/errors (players/markets skipped)"):
            for e in errors:
                st.write("- ", e)
    st.stop()

edges_df = pd.DataFrame(rows)

edge_cols = [c for c in edges_df.columns if c.startswith("edge_last_") and "7" not in c]
rate_cols = [c for c in edges_df.columns if c.startswith("over_rate_last_") and "7" not in c]

if not edge_cols or not rate_cols:
    st.error("Edge/over_rate columns missing from results.")
    st.stop()

edge_col = edge_cols[0]   # edge_last_N
rate_col = rate_cols[0]   # over_rate_last_N


# -------------------------------------------------
# EDGE TABLE WITH HIGHLIGHTING
# -------------------------------------------------
st.write("### Edges Table")

def highlight_edges(row):
    """Green for good edges, red for negative, yellow for small positive."""
    styles = [""] * len(row)
    try:
        val = float(row[edge_col])
        over = float(row[rate_col])
    except Exception:
        return styles

    # good edge: meets filters
    if val >= min_edge and over >= min_over_rate:
        color = "background-color: #d4f8d4"  # light green
    elif val < 0:
        color = "background-color: #f8d4d4"  # light red
    elif val > 0:
        color = "background-color: #fff4c2"  # light yellow
    else:
        return styles

    return [color] * len(row)

styled_edges = edges_df.sort_values(by=[rate_col, edge_col], ascending=False).style.apply(
    highlight_edges, axis=1
)

st.dataframe(styled_edges, use_container_width=True)

if errors:
    with st.expander("Show warnings/errors (players/markets skipped)"):
        for e in errors:
            st.write("- ", e)


# -------------------------------------------------
# AUTO BET SLIP BUILDER
# -------------------------------------------------
st.write("### Bet Slip Builder")

builder_df = edges_df.copy()
builder_df = builder_df.sort_values(by=[rate_col, edge_col], ascending=False).reset_index(drop=True)

# Add a checkbox column
builder_df.insert(0, "include", False)

edited = st.data_editor(
    builder_df,
    use_container_width=True,
    column_config={
        "include": st.column_config.CheckboxColumn(
            "Include", help="Check to add this prop to your slip."
        )
    },
    disabled=[c for c in builder_df.columns if c != "include"],
    key="bet_builder_editor",
)

selected = edited[edited["include"]]

if selected.empty:
    st.info("Select props in the table above to build a slip.")
else:
    st.success(f"{len(selected)} props selected for your slip.")
    st.dataframe(
        selected[
            ["player", "team", "opponent", "market", "line", "season_avg", edge_col, rate_col]
        ].reset_index(drop=True),
        use_container_width=True,
    )

    avg_edge = selected[edge_col].mean()
    avg_hit = selected[rate_col].mean()

    st.markdown(
        f"**Slip summary:** Avg edge = `{avg_edge:.2f}`, "
        f"Avg hit rate (last N) = `{avg_hit:.2f}`"
    )
    st.caption(
        "This does *not* place bets. Use it as a helper to choose legs before entering them on PrizePicks."
    )


# -------------------------------------------------
# PLAYER DETAIL
# -------------------------------------------------
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
