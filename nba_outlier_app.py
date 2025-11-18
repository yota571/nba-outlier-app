import time
from datetime import datetime

import pandas as pd
import streamlit as st

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll

from DFS_Wrapper import PrizePick, Underdog


# -------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="NBA Outlier-Style App",
    layout="wide",
)


# -------------------------------------------------
# HELPERS: CSV LOADER (OPTIONAL)
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

    # Tag as custom source
    df["book"] = "Custom"

    return df[required_cols + ["book"]]


# -------------------------------------------------
# FILTERS FOR FULL-GAME ONLY (NO 1H / 2H / QUARTERS / FIRST 5 MIN)
# -------------------------------------------------
def _is_full_game_prop_pp(item: dict, stat_type: str | None) -> bool:
    """
    PrizePicks: try to detect and exclude 1H / 2H / quarter / first 5-min props.
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
        "in 5 min", "in first six", "in first seven",
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


def _is_full_game_stat_ud(stat_type: str) -> bool:
    """
    Underdog: stat_type itself includes 1H / 1Q / First 5 Min / etc.
    Filter those out.
    """
    s = (stat_type or "").lower()

    bad_keywords = [
        "1h", " 1h ", "2h",
        "first half", "1st half", "second half", "2nd half",
        "half points", "half pts",
        "1q", "2q", "3q", "4q",
        "1st quarter", "2nd quarter", "3rd quarter", "4th quarter",
        "first quarter", "second quarter", "third quarter", "fourth quarter",
        "first 5", "in first 5", "first five",
        "first 3 min", "first 6 min", "first 7 min",
        "in first 3", "in first 6", "in first 7",
        "in first five",
        "quarter",
        "first half fantasy",
    ]

    for kw in bad_keywords:
        if kw in s:
            return False

    return True


# -------------------------------------------------
# PRIZEPICKS LOADER (NBA, FULL GAME, MAIN LINE ONLY)
# -------------------------------------------------
@st.cache_data
def load_prizepicks_nba_props() -> pd.DataFrame:
    """
    Pull cleaned NBA props from PrizePicks.

    - NBA only
    - pre-game only
    - full-game only (no halves/quarters/first 5 min)
    - keep ONLY one line per (book, player, team, opponent, market) – highest line
    """
    try:
        pp = PrizePick()
        raw = pp.get_data(organize_data=False)
    except TypeError:
        # Older / different signature fallback
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
        ["book", "player_name", "team", "opponent", "market"],
        as_index=False,
    ).last()

    df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df[["player_name", "team", "opponent", "market", "line", "game_time", "book"]]


# -------------------------------------------------
# UNDERDOG LOADER (NBA, FULL GAME, MAIN LINE ONLY) – FIXED
# -------------------------------------------------
@st.cache_data
def load_underdog_nba_props() -> pd.DataFrame:
    """
    Pull cleaned NBA props from Underdog.

    - NBA only
    - full-game only (filters on stat_type)
    - keep ONLY one line per (book, player, team, opponent, market) – highest line
    """
    try:
        ud = Underdog()
        # IMPORTANT: Underdog get_data() takes *no* arguments, returns non-organized list
        raw = ud.get_data()
    except Exception as e:
        st.error(f"Error loading Underdog data: {e}")
        return pd.DataFrame(
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
        )

    if not isinstance(raw, list):
        st.warning("Underdog data not in expected list format.")
        return pd.DataFrame(
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
        )

    records = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        sport_id = item.get("sport_id", "")
        if "NBA" not in str(sport_id).upper():
            continue

        player_name = item.get("player_name")
        if not player_name:
            continue

        team = item.get("team")
        opponent = item.get("opponent") or ""
        start_time = item.get("start_time") or None
        stats = item.get("stats") or []

        if not isinstance(stats, list):
            continue

        for stat in stats:
            stat_type = stat.get("stat_type")
            if not stat_type:
                continue

            if not _is_full_game_stat_ud(stat_type):
                continue

            line_val = stat.get("line_value")
            if line_val is None:
                continue

            try:
                line_val = float(line_val)
            except Exception:
                continue

            s = stat_type.strip()
            market_map_ud = {
                "Points": "points",
                "Rebounds": "rebounds",
                "Assists": "assists",
                "Pts + Rebs + Asts": "pra",
                "Pts+Rebs+Asts": "pra",
                "Points + Rebounds + Assists": "pra",
                "Rebounds + Assists": "ra",
                "Reb + Ast": "ra",
                "Reb+Ast": "ra",
                "3-Pointers Made": "threes",
                "Fantasy Points": "fs",
                "Fantasy Score": "fs",
            }
            market = market_map_ud.get(s)
            if market is None:
                continue

            if any(sep in opponent for sep in ("/", "|", "+")):
                continue

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
                    "book": "Underdog",
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
        ["book", "player_name", "team", "opponent", "market"],
        as_index=False,
    ).last()

    df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df[["player_name", "team", "opponent", "market", "line", "game_time", "book"]]


# -------------------------------------------------
# NBA STATS VIA nba_api
# -------------------------------------------------
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

    for p in all_players:
        if normalize_name(p["full_name"]) == target:
            return p["id"]

    for p in all_players:
        norm = normalize_name(p["full_name"])
        if target and (target in norm or norm in target):
            return p["id"]

    parts = target.split()
    if len(parts) > 1 and len(parts[0]) == 1:
        target_no_initial = " ".join(parts[1:])
        for p in all_players:
            norm = normalize_name(p["full_name"])
            if target_no_initial and (target_no_initial in norm or norm in target_no_initial):
                return p["id"]

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


# -------------------------------------------------
# SIDEBAR UI
# -------------------------------------------------
st.sidebar.title("NBA Outlier-Style App")

mode = st.sidebar.radio(
    "Prop source",
    ["PrizePicks only", "PrizePicks + Underdog", "Upload CSV manually"],
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
        "- This app uses **full-game props only** from PrizePicks and Underdog."
    )

st.sidebar.markdown(
    "Markets: **points, rebounds, assists, pra, ra (reb+ast), threes, fs (fantasy score)**"
)


# -------------------------------------------------
# LOAD PROPS (PP + UD + OPTIONAL CSV)
# -------------------------------------------------
props_df = None
pp_df = None
ud_df = None

if mode in ("PrizePicks only", "PrizePicks + Underdog"):
    pp_df = load_prizepicks_nba_props()
    if pp_df is None or pp_df.empty:
        st.warning("No PrizePicks props found.")

    if mode == "PrizePicks only":
        props_df = pp_df
        st.sidebar.success("Using PrizePicks NBA full-game props.")
    else:
        ud_df = load_underdog_nba_props()
        if ud_df is None or ud_df.empty:
            st.warning("No Underdog props found.")
            props_df = pp_df
        else:
            props_df = pd.concat([pp_df, ud_df], ignore_index=True)
        st.sidebar.success("Using PrizePicks + Underdog NBA full-game props.")
elif mode == "Upload CSV manually":
    uploaded = st.sidebar.file_uploader(
        "Upload props CSV",
        type=["csv"],
        help="Must contain columns: player_name, team, opponent, market, line, game_time",
    )
    if uploaded:
        props_df = load_props_from_csv(uploaded)

# Small debug display so you can confirm UD props are being picked up
with st.sidebar.expander("Source counts"):
    if pp_df is not None:
        st.write(f"PrizePicks props: {len(pp_df)}")
    if ud_df is not None:
        st.write(f"Underdog props: {len(ud_df)}")

if props_df is None or props_df.empty:
    st.info("No props loaded yet.")
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
books = ["All"] + sorted(props_df["book"].dropna().unique().tolist())

col1, col2, col3 = st.columns(3)
with col1:
    team_filter = st.selectbox("Team filter", teams)
with col2:
    market_filter = st.selectbox("Market filter", markets)
with col3:
    book_filter = st.selectbox("Book filter", books)

df = props_df.copy()

if team_filter != "All":
    df = df[df["team"] == team_filter]

if market_filter != "All":
    df = df[df["market"].str.lower() == market_filter.lower()]

if book_filter != "All":
    df = df[df["book"] == book_filter]

if df.empty:
    st.warning("No props match the selected filters.")
    st.stop()


# -------------------------------------------------
# EDGE / CONFIDENCE / PREDICTION / BET SIDE
# -------------------------------------------------
st.title("NBA Prop Edge Finder")
st.caption(
    "PrizePicks + Underdog full-game props vs nba_api stats → edges, confidence %, & line value."
)

st.write("### Calculating edges…")

rows = []
errors = []

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

    time.sleep(0.2)
    progress.progress((i + 1) / total_players)

progress.progress(1.0)
status_text.text("Computing edges, predictions, confidence, bet side…")

for _, row in df.iterrows():
    player_name = row.get("player_name")
    market = row.get("market")
    line = row.get("line")
    book = row.get("book") or "Unknown"

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


# -------------------------------------------------
# CARD VIEW (2 PER ROW) WITH "WHY THIS PROP?" BUTTON
# -------------------------------------------------
st.write("### Featured Edges (Card View)")

filtered_edges = edges_df[
    (edges_df[rate_col] >= min_over_rate)
    & (edges_df[edge_col] >= min_edge)
    & (edges_df["confidence_pct"] >= min_confidence)
]

if filtered_edges.empty:
    featured_df = edges_df.copy()
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
                    st.markdown(f"### {r['player']}  \n*{r['book']}*")
                    team = r.get("team") or ""
                    opp = r.get("opponent") or ""
                    st.markdown(f"**{team} vs {opp}**")
                    st.markdown(
                        f"*{r['market']}* &nbsp; | &nbsp; **Line:** `{r['line']}`"
                    )

                    side = r.get("bet_side", "No clear edge")
                    side_emoji = "⬆️" if side == "Over" else "⬇️" if side == "Under" else "⚖️"
                    st.markdown(f"**Recommended:** {side_emoji} **{side}**")

                    st.markdown(
                        f"Predicted: **{r['predicted_score']}**  "
                        f"(Season avg: {r['season_avg']})"
                    )

                    conf = r.get("confidence_pct", 0) or 0
                    edge_val = r.get(edge_col, 0) or 0
                    hit = r.get(rate_col, 0) or 0

                    st.markdown(
                        f"Edge: `{edge_val:.2f}` &nbsp; | &nbsp; "
                        f"Hit rate (N): `{hit:.2f}` &nbsp; | &nbsp; "
                        f"Confidence: `{conf:.1f}%`"
                    )
                    st.progress(min(max(conf / 100.0, 0.0), 1.0))

                    if st.button(
                        "Why this prop?",
                        key=f"why_{r['player']}_{r['market']}_{r['book']}_{k}",
                    ):
                        st.session_state["explain_row"] = r.to_dict()


# -------------------------------------------------
# BEST LINE VALUE BETWEEN BOOKS
# -------------------------------------------------
if mode == "PrizePicks + Underdog" and edges_df["book"].nunique() >= 2:
    st.write("### Best Line Value Between PrizePicks and Underdog")

    comp_rows = []

    grouped = edges_df.groupby(["player", "team", "opponent", "market"], dropna=False)

    for (player, team, opp, market), grp in grouped:
        if grp["book"].nunique() < 2:
            continue

        min_idx = grp["line"].idxmin()
        max_idx = grp["line"].idxmax()

        row_min = edges_df.loc[min_idx]
        row_max = edges_df.loc[max_idx]

        diff = row_max["line"] - row_min["line"]
        if diff <= 0:
            continue

        comp_rows.append(
            {
                "player": player,
                "team": team,
                "opponent": opp,
                "market": market,
                "min_line": row_min["line"],
                "min_line_book": row_min["book"],
                "max_line": row_max["line"],
                "max_line_book": row_max["book"],
                "line_diff": round(diff, 2),
                "best_book_for_over": row_min["book"],
                "best_book_for_under": row_max["book"],
            }
        )

    if not comp_rows:
        st.info("No player/market combos where lines differ between PrizePicks and Underdog.")
    else:
        comp_df = pd.DataFrame(comp_rows).sort_values("line_diff", ascending=False)
        st.caption(
            "For **Over** you generally want the **lower line**, for **Under** you want the **higher line**."
        )
        st.dataframe(comp_df, use_container_width=True)


# -------------------------------------------------
# EXPLANATION PANEL – WHY THIS PREDICTION & CONFIDENCE
# -------------------------------------------------
st.write("### Prop Explanation (Why this prediction & confidence)")

if "explain_row" in st.session_state:
    er = st.session_state["explain_row"]

    player_name = er.get("player")
    market = er.get("market")
    line = er.get("line")
    book = er.get("book", "Unknown")

    st.markdown(
        f"**Player:** {player_name}  \n"
        f"**Book:** {book}  \n"
        f"**Market:** {market}  \n"
        f"**Line:** `{line}`"
    )

    mask = (
        (edges_df["player"] == player_name)
        & (edges_df["market"] == market)
        & (edges_df["book"] == book)
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
            st.info("Not enough recent games to show detailed game-by-game history.")
else:
    st.caption("Click **“Why this prop?”** on any card above to see a detailed explanation here.")


# -------------------------------------------------
# FULL EDGES TABLE
# -------------------------------------------------
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
        color = "background-color: #d4f8d4"
    elif val < 0:
        color = "background-color: #f8d4d4"
    elif val > 0:
        color = "background-color: #fff4c2"
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

table_df = edges_df.copy()
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


# -------------------------------------------------
# PLAYER DETAIL (RAW GAME LOG)
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
