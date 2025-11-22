# =====================================================
# NBA Prop Edge Finder – PrizePicks
# Version 2.4.0 (Power Index + Live Odds API)
#
# - Based on v2.3 Power Index Edition
# - Adds integration with The Odds API for live:
#     * Moneyline odds
#     * Spreads
# - Blends:
#     * Prop-based game model
#     * Team Power Index model (OffRtg, DefRtg, NetRtg, Pace, W/L)
# - Then compares our model vs book lines:
#     * ml_edge_pct       = model win% - book implied win%
#     * spread_edge_pts   = model spread - book spread
# =====================================================

import math
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib  # ML model loader
import requests  # <-- added for The Odds API

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll

from DFS_Wrapper import PrizePick


# =====================================================
# CONFIG
# =====================================================

BETS_FILE = "bet_tracker.csv"
ML_MODEL_PATH = "over_model.pkl"  # <-- your trained ML model file
THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"


# =====================================================
# BET PERSISTENCE HELPERS
# =====================================================

def load_bets_from_disk() -> list:
    """Load saved bets from CSV if it exists."""
    if not os.path.exists(BETS_FILE):
        return []
    try:
        df = pd.read_csv(BETS_FILE)
        return df.to_dict(orient="records")
    except Exception:
        return []


def save_bets_to_disk(bets: list) -> None:
    """Persist bets to CSV."""
    try:
        df = pd.DataFrame(bets)
        df.to_csv(BETS_FILE, index=False)
    except Exception:
        pass


@st.cache_resource
def load_over_model():
    """Load the ML model from disk if available."""
    try:
        model = joblib.load(ML_MODEL_PATH)
        return model
    except Exception:
        return None


over_model = load_over_model()


# =====================================================
# STREAMLIT PAGE CONFIG & STATE
# =====================================================
st.set_page_config(
    page_title="NBA Outlier-Style App (PrizePicks, Power Index + Odds)",
    layout="wide",
)

# persistent bet tracker backed by CSV
if "bet_tracker" not in st.session_state:
    st.session_state["bet_tracker"] = load_bets_from_disk()


# =====================================================
# HELPERS
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


def prob_to_american(p: float) -> str:
    """Convert probability (0–1) to American odds string."""
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


def american_to_prob(odds) -> float | None:
    """Convert American odds to implied probability (0–1)."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


# =====================================================
# PRIZEPICKS LOADER (NBA, FULL GAME ONLY)
# =====================================================
@st.cache_data
def load_prizepicks_nba_props() -> pd.DataFrame:
    """
    Pull cleaned NBA props from PrizePicks.

    - NBA only
    - active/open markets
    - full-game only
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

        # status: keep anything that isn't clearly closed/settled/void
        status = str(item.get("status", "")).lower()
        bad_statuses = [
            "settled", "graded", "closed", "void", "canceled",
            "cancelled", "refunded", "suspended"
        ]
        if any(bad in status for bad in bad_statuses):
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


@st.cache_data
def get_all_teams():
    """Fetch NBA teams from nba_api (logo/metadata)."""
    return teams.get_teams()


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    for ch in [".", ",", "'", "`"]:
        name = name.replace(ch, "")
    name = " ".join(name.split())
    return name


def map_team_to_abbrev(team_name: str | None) -> str | None:
    """
    Map a generic team string (LAL, Lakers, Los Angeles Lakers) to official NBA abbreviation.
    """
    if not team_name:
        return None

    raw = str(team_name).strip()
    if not raw:
        return None

    all_teams = get_all_teams()
    lower_raw = raw.lower()

    # Looks like an abbreviation already
    if len(raw) <= 3:
        cand = raw.upper()
        for t in all_teams:
            if t.get("abbreviation", "").upper() == cand:
                return cand

    # Exact full_name match
    for t in all_teams:
        if lower_raw == str(t.get("full_name", "")).lower():
            return t.get("abbreviation")

    # Exact nickname match (e.g. "Lakers")
    for t in all_teams:
        if lower_raw == str(t.get("nickname", "")).lower():
            return t.get("abbreviation")

    # Contained in full_name ("los angeles lakers" contains "lakers")
    for t in all_teams:
        if lower_raw in str(t.get("full_name", "")).lower():
            return t.get("abbreviation")

    # Fallback: uppercase short code
    if len(raw) <= 3:
        return raw.upper()
    return None


def get_team_logo_url(team_name: str | None) -> str | None:
    """
    Map team string (like 'LAL', 'Lakers', 'Los Angeles Lakers') to NBA logo URL.
    """
    abbr = map_team_to_abbrev(team_name)
    if not abbr:
        return None

    try:
        all_teams = get_all_teams()
    except Exception:
        return None

    for t in all_teams:
        if str(t.get("abbreviation", "")).upper() == abbr:
            return f"https://cdn.nba.com/logos/nba/{t['id']}/primary/L/logo.svg"

    return None


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


# =====================================================
# TEAM ADVANCED STATS & POWER INDEX
# =====================================================
@st.cache_data
def get_team_advanced_stats() -> pd.DataFrame:
    """
    Pull team advanced stats from NBA API and build a Power Index.
    Uses OffRtg, DefRtg, NetRtg, Pace, W_PCT.
    """
    try:
        from nba_api.stats.endpoints import leaguedashteamstats

        lds = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_def="Advanced",
            per_mode_detailed="PerGame",
        )
        df = lds.get_data_frames()[0]

        all_teams = get_all_teams()
        id_to_abbrev = {t["id"]: t["abbreviation"] for t in all_teams}
        df["TEAM_ABBREV"] = df["TEAM_ID"].map(id_to_abbrev)

        # Ensure required columns
        for col in ["OFF_RATING", "DEF_RATING", "NET_RATING", "W_PCT", "PACE"]:
            if col not in df.columns:
                df[col] = 0.0

        # Center pace so it doesn't blow up power
        pace_mean = df["PACE"].mean() if not df["PACE"].isna().all() else 0.0

        # Simple power formula
        df["POWER"] = (
            df["NET_RATING"]
            + (df["W_PCT"] - 0.5) * 20.0
            + (df["OFF_RATING"] - df["DEF_RATING"]) * 0.25
            + (df["PACE"] - pace_mean) * 0.1
        )

        return df
    except Exception:
        return pd.DataFrame()


# =====================================================
# LIVE ODDS FROM THE ODDS API
# =====================================================
@st.cache_data
def load_odds_from_the_odds_api(api_key: str) -> pd.DataFrame:
    """
    Fetch current NBA odds (h2h + spreads) from The Odds API.
    Returns one row per (game_key, team_abbrev) with:
      - book_ml_odds
      - book_spread
      - book_spread_odds
    """
    if not api_key:
        return pd.DataFrame()

    try:
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,spreads",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        resp = requests.get(THE_ODDS_API_URL, params=params, timeout=10)
        if resp.status_code != 200:
            st.warning(f"The Odds API error: {resp.status_code} – {resp.text}")
            return pd.DataFrame()

        data = resp.json()
        rows = []

        for game in data:
            home = game.get("home_team")
            away = game.get("away_team")
            commence = game.get("commence_time")
            sites = game.get("bookmakers", [])

            if not home or not away or not sites:
                continue

            home_abbrev = map_team_to_abbrev(home)
            away_abbrev = map_team_to_abbrev(away)
            # Use abbrevs if we have them; otherwise names
            t1 = home_abbrev or home
            t2 = away_abbrev or away
            teams_sorted = sorted([t1, t2])
            game_key = " vs ".join(teams_sorted)

            # For now pick the first book (or you can filter for a specific one like "draftkings")
            site = sites[0]
            site_key = site.get("key")

            h2h = None
            spreads = None
            for m in site.get("markets", []):
                if m.get("key") == "h2h":
                    h2h = m
                elif m.get("key") == "spreads":
                    spreads = m

            for team_name in [home, away]:
                team_abbrev = map_team_to_abbrev(team_name)

                ml_odds = None
                spread_pt = None
                spread_odds = None

                if h2h:
                    for o in h2h.get("outcomes", []):
                        if o.get("name") == team_name:
                            ml_odds = o.get("price")

                if spreads:
                    for o in spreads.get("outcomes", []):
                        if o.get("name") == team_name:
                            spread_pt = o.get("point")
                            spread_odds = o.get("price")

                rows.append(
                    {
                        "game_key": game_key,
                        "team_name_api": team_name,
                        "team_abbrev": team_abbrev,
                        "book_key": site_key,
                        "book_ml_odds": ml_odds,
                        "book_spread": spread_pt,
                        "book_spread_odds": spread_odds,
                        "commence_time": commence,
                    }
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        st.warning(f"Error loading The Odds API data: {e}")
        return pd.DataFrame()


# =====================================================
# SIDEBAR UI
# =====================================================
st.sidebar.title("NBA Outlier-Style App – v2.4 Odds")

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

# ML-style team model controls
min_props_for_ml = st.sidebar.slider(
    "Min props per team for moneyline/spread model", min_value=1, max_value=20, value=3, step=1
)

spread_highlight = st.sidebar.slider(
    "Min |model spread| to flag a game",
    min_value=0.0,
    max_value=15.0,
    value=4.5,
    step=0.5,
)

ml_winprob_highlight = st.sidebar.slider(
    "Min win % to flag ML favorite",
    min_value=50.0,
    max_value=80.0,
    value=60.0,
    step=1.0,
)

# Blend weight: 0 = props only, 1 = power index only
power_weight = st.sidebar.slider(
    "Weight for team Power Index in ML model",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

# The Odds API integration
use_odds_api = st.sidebar.checkbox("Use live book odds (The Odds API)", value=False)
if use_odds_api:
    odds_api_key = st.sidebar.text_input("The Odds API key", type="password")
else:
    odds_api_key = ""

# Mobile layout toggle
mobile_mode = st.sidebar.checkbox("Mobile mode (simplified layout)", value=False)

st.sidebar.markdown("### Quick filter preset")
preset = st.sidebar.selectbox(
    "Preset (optional)",
    ["Custom", "High Edge", "High Confidence", "Loose"],
)

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

if over_model is None:
    st.sidebar.info(
        "ML file 'over_model.pkl' not found – using fallback ML-style probabilities for Over."
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

# safer "only today" filter: keep NaT rows, only drop clearly non-today games
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
teams_list = ["All teams"] + sorted(props_df["team"].dropna().unique().tolist())
markets = ["All markets"] + sorted(props_df["market"].dropna().unique().tolist())
games_list = ["All games"] + sorted(props_df["game_label"].dropna().unique().tolist())

top_col1, top_col2, top_col3, top_col4 = st.columns([2, 2, 3, 3])
with top_col1:
    team_filter = st.selectbox("Team filter", teams_list)
with top_col2:
    market_filter = st.selectbox("Market filter (points, pra, etc)", markets)
with top_col3:
    game_filter = st.selectbox("Game filter", games_list)
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
# EDGE / CONFIDENCE / PREDICTION / BET SIDE / ODDS / ML
# =====================================================
st.title("NBA Prop Edge Finder – v2.4 Power Index + Live Odds")

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
status_text.text("Computing edges & ML-style probabilities…")


def build_ml_features(
    season_avg: float,
    avg_last_n: float,
    edge_n: float,
    over_rate_n: float,
    last_game_stat: float,
    line_float: float,
    is_home: int,
    days_rest: int,
) -> dict:
    return {
        "season_avg": season_avg,
        "last_n_avg": avg_last_n,
        "edge_last_n": edge_n,
        "over_rate_last_n": over_rate_n,
        "last_game_stat": last_game_stat,
        "line": line_float,
        "line_minus_season": line_float - season_avg,
        "line_minus_last_n": line_float - avg_last_n,
        "is_home": is_home,
        "days_rest": days_rest,
    }


ML_FEATURE_COLS = [
    "season_avg",
    "last_n_avg",
    "edge_last_n",
    "over_rate_last_n",
    "last_game_stat",
    "line",
    "line_minus_season",
    "line_minus_last_n",
    "is_home",
    "days_rest",
]

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

    # blended prediction (historical-based)
    w_season = 0.4
    w_last_n = 0.4
    w_last_7 = 0.2

    avg7_for_blend = avg_last_7 if avg_last_7 is not None else avg_last_n
    predicted_score = (
        w_season * season_avg
        + w_last_n * avg_last_n
        + w_last_7 * avg7_for_blend
    )

    # Confidence based on historical only
    hit_score = over_rate_n
    if line_float != 0:
        edge_ratio = max(0.0, edge_n / max(1.0, line_float))
    else:
        edge_ratio = 0.0
    edge_score = max(0.0, min(1.0, edge_ratio * 4.0))
    confidence = 0.6 * hit_score + 0.4 * edge_score
    confidence_pct = round(confidence * 100, 1)

    # Historical-based probabilities and odds
    over_prob_hist = float(over_rate_n)
    over_prob_hist_clamped = min(max(over_prob_hist, 0.01), 0.99)
    under_prob_hist = 1.0 - over_prob_hist_clamped
    over_odds_hist = prob_to_american(over_prob_hist_clamped)
    under_odds_hist = prob_to_american(under_prob_hist)

    # Determine bet side
    delta = predicted_score - line_float
    if delta >= 0.5:
        bet_side = "Over"
    elif delta <= -0.5:
        bet_side = "Under"
    else:
        bet_side = "No clear edge"

    if bet_side == "Over":
        bet_odds = over_odds_hist
    elif bet_side == "Under":
        bet_odds = under_odds_hist
    else:
        bet_odds = "N/A"

    # =====================
    # ML PROBABILITY OF OVER
    # =====================
    ml_prob_over = None
    ml_odds_over = None

    try:
        last_game_stat = float(series.iloc[0]) if len(series) > 0 else avg_last_n

        if "IS_AWAY" in gamelog.columns and len(gamelog) > 0:
            is_away = bool(gamelog["IS_AWAY"].iloc[0])
            is_home = 0 if is_away else 1
        else:
            is_home = 1

        if len(gamelog) >= 2:
            days_rest = (gamelog["GAME_DATE"].iloc[0] - gamelog["GAME_DATE"].iloc[1]).days
            days_rest = max(days_rest, 0)
        else:
            days_rest = 2

        if over_model is not None:
            feat_dict = build_ml_features(
                season_avg=season_avg,
                avg_last_n=avg_last_n,
                edge_n=edge_n,
                over_rate_n=over_rate_n,
                last_game_stat=last_game_stat,
                line_float=line_float,
                is_home=is_home,
                days_rest=days_rest,
            )
            X_row = [[feat_dict[c] for c in ML_FEATURE_COLS]]
            prob = float(over_model.predict_proba(X_row)[0][1])  # class 1 = Over
            prob = max(0.001, min(0.999, prob))
            ml_prob_over = prob
            ml_odds_over = prob_to_american(prob)
        else:
            # fallback "ML-style" probability when no over_model.pkl
            edge_scaled = max(-5.0, min(5.0, edge_n / max(1.0, line_float) * 4.0))
            base_logit = math.log(over_prob_hist_clamped / (1 - over_prob_hist_clamped))
            blended_logit = base_logit + 0.8 * edge_scaled
            prob = 1.0 / (1.0 + math.exp(-blended_logit))
            prob = max(0.001, min(0.999, prob))
            ml_prob_over = prob
            ml_odds_over = prob_to_american(prob)
    except Exception:
        pass

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
            "over_prob": round(over_prob_hist_clamped, 3),
            "under_prob": round(under_prob_hist, 3),
            "over_odds": over_odds_hist,
            "under_odds": under_odds_hist,
            "bet_odds": bet_odds,
            "ml_prob_over": round(ml_prob_over, 3) if ml_prob_over is not None else None,
            "ml_odds_over": ml_odds_over,
            "game_time": row.get("game_time"),
            "game_label": row.get("game_label"),
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

# Add mapped team abbreviations for power + odds model
edges_df["team_abbrev"] = edges_df["team"].apply(map_team_to_abbrev)
edges_df["opponent_abbrev"] = edges_df["opponent"].apply(map_team_to_abbrev)


# =====================================================
# BUILD GAME MONEYLINE PREDICTIONS + MODEL SPREADS (PROPS + POWER + ODDS)
# =====================================================
def build_moneyline_predictions(
    edges,
    team_stats_df=None,
    odds_df=None,
    scoring_markets=None,
    power_weight: float = 0.5,
):
    if scoring_markets is None:
        scoring_markets = ["points", "pra", "fs", "threes"]

    if edges is None or edges.empty:
        return pd.DataFrame()

    required_cols = {"team", "opponent", "market", edge_col, "confidence_pct", "bet_side"}
    if not required_cols.issubset(edges.columns):
        return pd.DataFrame()

    df_ml = edges.copy()
    df_ml = df_ml[df_ml["market"].isin(scoring_markets)]
    if df_ml.empty:
        return pd.DataFrame()

    def make_game_key(row):
        t = str(row.get("team", "NA"))
        o = str(row.get("opponent", "NA"))
        teams_sorted = sorted([t, o])
        return " vs ".join(teams_sorted)

    df_ml["game_key"] = df_ml.apply(make_game_key, axis=1)

    def row_signal(row):
        side = str(row.get("bet_side", ""))
        try:
            e_val = float(row[edge_col])
            conf = float(row.get("confidence_pct", 0)) / 100.0
        except Exception:
            return 0.0
        base = e_val * conf
        if side == "Over":
            return base
        elif side == "Under":
            return -base
        else:
            return 0.0

    df_ml["signal"] = df_ml.apply(row_signal, axis=1)

    grp = df_ml.groupby(
        ["game_key", "team", "opponent"],
        as_index=False,
    ).agg(
        props_count=("signal", "size"),
        avg_confidence=("confidence_pct", "mean"),
        avg_edge=(edge_col, "mean"),
        avg_signal=("signal", "mean"),
    )

    if grp.empty:
        return grp

    # ---- Props-based probability & spread (original model) ----
    def to_prob(sig):
        try:
            return 1.0 / (1.0 + math.exp(-0.8 * sig))
        except OverflowError:
            return 1.0 if sig > 0 else 0.0

    grp["raw_prob"] = grp["avg_signal"].apply(to_prob)
    grp["sum_prob_game"] = grp.groupby("game_key")["raw_prob"].transform("sum")

    def norm_prob(row):
        s = row["sum_prob_game"]
        rp = row["raw_prob"]
        if s <= 0:
            return 0.5
        p = rp / s
        return max(0.05, min(0.95, p))

    grp["win_prob"] = grp.apply(norm_prob, axis=1)
    grp["ml_odds"] = grp["win_prob"].apply(prob_to_american)
    grp["win_prob_pct"] = (grp["win_prob"] * 100).round(1)

    grp["signal_centered"] = grp["avg_signal"] - grp.groupby("game_key")["avg_signal"].transform("mean")
    spread_factor = 7.0
    grp["model_spread"] = (grp["signal_centered"] * spread_factor).round(1)
    grp["game_label"] = grp.apply(
        lambda r: f"{r['team']} vs {r['opponent']}", axis=1
    )

    # ---- If no team stats, just use props-only model ----
    if team_stats_df is None or team_stats_df.empty:
        grp["OFF_RATING"] = None
        grp["DEF_RATING"] = None
        grp["NET_RATING"] = None
        grp["PACE"] = None
        grp["W"] = None
        grp["L"] = None
        grp["W_PCT"] = None
        grp["POWER"] = None
        grp["power_prob"] = None
        grp["power_ml_odds"] = None
        grp["power_win_pct"] = None
        grp["power_spread"] = None

        grp["final_win_prob"] = grp["win_prob"]
        grp["final_ml_odds"] = grp["ml_odds"]
        grp["final_win_prob_pct"] = grp["win_prob_pct"]
        grp["final_model_spread"] = grp["model_spread"]

        # Odds integration (if provided)
        grp["team_abbrev"] = grp["team"].apply(map_team_to_abbrev)
        if odds_df is not None and not odds_df.empty:
            merge_cols = ["game_key", "team_abbrev"]
            for c in ["game_key", "team_abbrev"]:
                if c not in odds_df.columns:
                    odds_df[c] = None
            grp = grp.merge(
                odds_df,
                on=["game_key", "team_abbrev"],
                how="left",
            )

            grp["book_ml_prob"] = grp["book_ml_odds"].apply(american_to_prob)
            grp["ml_edge_pct"] = (grp["final_win_prob"] - grp["book_ml_prob"]).where(
                grp["book_ml_prob"].notna()
            ) * 100.0
            grp["spread_edge_pts"] = (grp["final_model_spread"] - grp["book_spread"]).where(
                grp["book_spread"].notna()
            )
        else:
            grp["book_ml_odds"] = None
            grp["book_spread"] = None
            grp["book_spread_odds"] = None
            grp["book_ml_prob"] = None
            grp["ml_edge_pct"] = None
            grp["spread_edge_pts"] = None

        grp = grp.sort_values(["final_win_prob", "game_key"], ascending=[False, True])
        return grp

    # ---- Merge in team stats / power index ----
    grp["team_abbrev"] = grp["team"].apply(map_team_to_abbrev)

    merge_cols = [
        "TEAM_ABBREV",
        "TEAM_NAME",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "PACE",
        "W",
        "L",
        "W_PCT",
        "POWER",
    ]

    ts = team_stats_df.copy()
    for c in merge_cols:
        if c not in ts.columns:
            ts[c] = None

    grp = grp.merge(
        ts[merge_cols],
        left_on="team_abbrev",
        right_on="TEAM_ABBREV",
        how="left",
    )

    # ---- Power-based probabilities and spreads ----
    grp["POWER_SAFE"] = grp["POWER"].fillna(grp["POWER"].mean() if not grp["POWER"].isna().all() else 0.0)

    def game_power_softmax(sub_df):
        vals = sub_df["POWER_SAFE"].values
        if len(vals) == 0:
            return np.array([])
        vals_centered = vals - np.mean(vals)
        exps = np.exp(0.15 * vals_centered)
        if exps.sum() <= 0:
            return np.ones_like(exps) / len(exps)
        return exps / exps.sum()

    grp["power_prob"] = grp.groupby("game_key", group_keys=False).apply(
        lambda g: pd.Series(game_power_softmax(g), index=g.index)
    )

    grp["power_prob"] = grp["power_prob"].clip(0.05, 0.95)
    grp["power_ml_odds"] = grp["power_prob"].apply(prob_to_american)
    grp["power_win_pct"] = (grp["power_prob"] * 100).round(1)

    grp["power_centered"] = grp["POWER_SAFE"] - grp.groupby("game_key")["POWER_SAFE"].transform("mean")
    power_spread_factor = 6.0
    grp["power_spread"] = (grp["power_centered"] * power_spread_factor).round(1)

    # ---- Blend props model + power model ----
    pw = max(0.0, min(1.0, float(power_weight)))

    grp["final_win_prob"] = (1.0 - pw) * grp["win_prob"] + pw * grp["power_prob"]
    grp["final_ml_odds"] = grp["final_win_prob"].apply(prob_to_american)
    grp["final_win_prob_pct"] = (grp["final_win_prob"] * 100).round(1)

    grp["final_model_spread"] = (
        (1.0 - pw) * grp["model_spread"] + pw * grp["power_spread"]
    ).round(1)

    # ---- Merge in live odds from The Odds API (if any) ----
    if odds_df is not None and not odds_df.empty:
        for c in ["game_key", "team_abbrev"]:
            if c not in odds_df.columns:
                odds_df[c] = None

        grp = grp.merge(
            odds_df,
            on=["game_key", "team_abbrev"],
            how="left",
        )

        grp["book_ml_prob"] = grp["book_ml_odds"].apply(american_to_prob)
        grp["ml_edge_pct"] = (grp["final_win_prob"] - grp["book_ml_prob"]).where(
            grp["book_ml_prob"].notna()
        ) * 100.0
        grp["spread_edge_pts"] = (grp["final_model_spread"] - grp["book_spread"]).where(
            grp["book_spread"].notna()
        )
    else:
        grp["book_ml_odds"] = None
        grp["book_spread"] = None
        grp["book_spread_odds"] = None
        grp["book_ml_prob"] = None
        grp["ml_edge_pct"] = None
        grp["spread_edge_pts"] = None

    grp = grp.sort_values(["final_win_prob", "game_key"], ascending=[False, True])

    return grp


team_stats_df = get_team_advanced_stats()

if use_odds_api and odds_api_key:
    odds_df = load_odds_from_the_odds_api(odds_api_key)
else:
    odds_df = pd.DataFrame()

games_ml_df = build_moneyline_predictions(
    edges_df,
    team_stats_df=team_stats_df,
    odds_df=odds_df,
    power_weight=power_weight,
)


# =====================================================
# TABS: CARDS / TABLE / PLAYER DETAIL / GAME ML / BETS
# =====================================================
tab_cards, tab_table, tab_player, tab_games, tab_bets = st.tabs(
    ["Cards & Explanation", "Table", "Player Detail", "Game Moneylines & Spreads (MODEL)", "Bet Tracker (ML & learning)"]
)


# =====================================================
# TAB 1: CARD VIEW
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
        def render_player_card(r, k_suffix: str):
            card = st.container()
            with card:
                top_col1, top_col2 = st.columns([1, 2])

                with top_col1:
                    pid = r.get("player_id")
                    if pid:
                        try:
                            img_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{int(pid)}.png"
                            st.image(img_url)
                        except Exception:
                            st.write(" ")
                    else:
                        st.write(" ")

                with top_col2:
                    st.markdown(f"### {r['player']}")
                    team_val = r.get("team") or ""
                    opp = r.get("opponent") or ""
                    st.markdown(f"**{team_val} vs {opp}**")
                    st.markdown(
                        f"*{r['market']}* | Line: `{r['line']}`"
                    )

                    side = r.get("bet_side", "No clear edge")
                    if side == "Over":
                        side_emoji = "⬆️"
                    elif side == "Under":
                        side_emoji = "⬇️"
                    else:
                        side_emoji = "⚖️"
                    st.markdown(f"Recommended: {side_emoji} **{side}**")

                    st.markdown(
                        f"Predicted: **{r['predicted_score']}**  "
                        f"(Season avg: {r['season_avg']})"
                    )

                conf = r.get("confidence_pct", 0) or 0
                edge_val = r.get(edge_col, 0) or 0
                hit = r.get(rate_col, 0) or 0

                st.markdown(
                    f"Edge: `{edge_val:.2f}` | "
                    f"Hit rate (N): `{hit:.2f}` | "
                    f"Confidence: `{conf:.1f}%`"
                )
                st.progress(min(max(conf / 100.0, 0.0), 1.0))

                side = r.get("bet_side", "No clear edge")
                if side == "Over":
                    hist_prob = r.get("over_prob", 0)
                    hist_odds = r.get("over_odds", "N/A")
                elif side == "Under":
                    hist_prob = r.get("under_prob", 0)
                    hist_odds = r.get("under_odds", "N/A")
                else:
                    hist_prob = None
                    hist_odds = "N/A"

                if hist_prob is not None:
                    st.markdown(
                        f"Historical {side} odds: `{hist_odds}` "
                        f"({hist_prob*100:.1f}% implied)"
                    )

                ml_prob = r.get("ml_prob_over")
                ml_odds = r.get("ml_odds_over")
                if ml_prob is not None and ml_odds is not None:
                    st.markdown(
                        f"ML Over prob: `{ml_odds}`  "
                        f"({ml_prob*100:.1f}% from model)"
                    )

                if st.button(
                    "Why this prop?",
                    key=f"why_{r['player']}_{r['market']}_{k_suffix}",
                ):
                    st.session_state["explain_row"] = r.to_dict()

                if st.button(
                    "Track this bet",
                    key=f"track_{r['player']}_{r['market']}_{k_suffix}",
                ):
                    bet = {
                        "bet_category": "player_prop",
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
                        "game_key": r.get("game_label"),
                        "model_spread": None,
                        "win_prob_pct": None,
                        "actual_stat": None,
                        "result": None,
                    }
                    st.session_state["bet_tracker"].append(bet)
                    save_bets_to_disk(st.session_state["bet_tracker"])
                    st.success("Bet added to tracker for this prop.")

        if mobile_mode:
            for k in range(top_n):
                r = featured_df.iloc[k]
                render_player_card(r, f"m_{k}")
        else:
            for idx in range(0, top_n, 2):
                cols = st.columns(2)
                for j in range(2):
                    k = idx + j
                    if k >= top_n:
                        break
                    r = featured_df.iloc[k]
                    with cols[j]:
                        render_player_card(r, f"d_{k}")

    st.write("### Prop Explanation")

    if "explain_row" in st.session_state:
        er = st.session_state["explain_row"]

        player_name = er.get("player")
        market = er.get("market")
        line = er.get("line")

        st.markdown(
            f"Player: **{player_name}**  \n"
            f"Market: **{market}**  \n"
            f"Line: `{line}`"
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
            ml_prob = r.get("ml_prob_over")
            ml_odds = r.get("ml_odds_over")

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("Key Numbers")
                st.write(f"- Season average: {season_avg}")
                st.write(f"- Last {games_to_look_back} avg: {avg_last_n}")
                st.write(f"- Hit rate last {games_to_look_back}: {over_rate_n:.2f}")
                st.write(f"- Edge last {games_to_look_back}: {edge_n:.2f} vs line {line}")
                if avg_last_7 is not None:
                    st.write(f"- Last 7 avg: {avg_last_7}")
                    st.write(f"- Hit rate last 7: {(over_rate_7 or 0):.2f}")
                    st.write(f"- Edge last 7: {(edge_7 or 0):.2f}")

            with col_right:
                st.markdown("Model Result")
                st.write(f"- Predicted score: {predicted_score}")
                st.write(f"- Confidence: {confidence_pct:.1f}%")
                st.write(f"- Historical Over odds: {over_odds}")
                st.write(f"- Historical Under odds: {under_odds}")
                if ml_prob is not None and ml_odds is not None:
                    st.write(f"- ML Over probability: {ml_prob*100:.1f}% ({ml_odds})")
                else:
                    st.write("- ML Over probability: not available")
                st.write(f"- Recommended side: {bet_side}  (odds: {bet_odds})")

    else:
        st.caption("Click 'Why this prop?' on any card above to see a detailed explanation here.")


# =====================================================
# TAB 2: FULL EDGES TABLE
# =====================================================
with tab_table:
    st.write("### Full Edges Table")

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
        "ml_prob_over",
        "ml_odds_over",
        "confidence_pct",
        "game_time",
    ]

    table_df = view_df.copy()
    for col in display_cols:
        if col not in table_df.columns:
            table_df[col] = None

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

        if mobile_mode:
            with st.expander("Show full edges table"):
                st.dataframe(styled_edges, use_container_width=True)
        else:
            st.dataframe(styled_edges, use_container_width=True)

    if errors:
        with st.expander("Warnings/errors (players/markets skipped)"):
            for e in errors:
                st.write("- ", e)


# =====================================================
# TAB 3: PLAYER DETAIL
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
# TAB 4: GAME MONEYLINES & SPREADS (MODEL + ODDS)
# =====================================================
with tab_games:
    st.write("### Game Moneylines & Model Spreads (Props + Power Index + Book Odds)")

    if use_odds_api and odds_df.empty:
        st.info("The Odds API is enabled but no odds were returned. Check your API key / usage.")

    if games_ml_df.empty:
        st.info(
            "Not enough scoring-related props to build game-level moneyline/spread predictions yet.\n\n"
            "We use points / PRA / threes / fantasy score props as the main signal, "
            "blended with team Power Index and compared against book odds when available."
        )
    else:
        ml_df = games_ml_df.copy()
        ml_df = ml_df[ml_df["props_count"] >= min_props_for_ml]

        if ml_df.empty:
            st.warning(
                f"No teams meet the minimum of {min_props_for_ml} props for a moneyline/spread prediction."
            )
        else:
            favorites = (
                ml_df.sort_values("final_win_prob", ascending=False)
                .groupby("game_key")
                .head(1)
                .sort_values("final_win_prob", ascending=False)
            )

            fav_list = favorites.to_dict(orient="records")

            st.markdown("#### Game Cards (favorite vs underdog)")

            if not fav_list:
                st.info("No games available for card view.")
            else:
                def render_game_card(fav, k_suffix: str):
                    game_key = fav["game_key"]
                    fav_team = fav["team"]
                    opp_team = fav["opponent"]

                    dog_row = ml_df[
                        (ml_df["game_key"] == game_key)
                        & (ml_df["team"] != fav_team)
                    ]
                    dog = dog_row.to_dict(orient="records")[0] if not dog_row.empty else None

                    card = st.container()
                    with card:
                        st.markdown(f"### {fav.get('game_label', game_key)}")

                        logo_c1, logo_c2 = st.columns(2)
                        with logo_c1:
                            fav_logo = get_team_logo_url(fav_team)
                            if fav_logo:
                                st.image(fav_logo, width=80)
                        with logo_c2:
                            if dog:
                                dog_logo = get_team_logo_url(dog["team"])
                                if dog_logo:
                                    st.image(dog_logo, width=80)

                        top_c1, top_c2 = st.columns(2)
                        with top_c1:
                            st.markdown("**Favorite (blended model)**")
                            st.markdown(f"Team: **{fav_team}**")

                            wp_final = fav.get("final_win_prob_pct")
                            wp_props = fav.get("win_prob_pct")
                            wp_power = fav.get("power_win_pct")
                            final_odds = fav.get("final_ml_odds")
                            props_odds = fav.get("ml_odds")
                            power_odds = fav.get("power_ml_odds")

                            if wp_final is not None:
                                st.markdown(
                                    f"- Final win %: **{wp_final:.1f}%**  "
                                    f"({final_odds})"
                                )
                            if wp_props is not None:
                                st.markdown(
                                    f"- Props model: {wp_props:.1f}%  ({props_odds})"
                                )
                            if wp_power is not None:
                                st.markdown(
                                    f"- Power idx: {wp_power:.1f}%  ({power_odds})"
                                )

                            sp_final = fav.get("final_model_spread")
                            sp_props = fav.get("model_spread")
                            sp_power = fav.get("power_spread")

                            if sp_final is not None:
                                st.markdown(
                                    f"- Final model spread: **{sp_final:+.1f}**"
                                )
                            if sp_props is not None:
                                st.markdown(
                                    f"- Props-only spread: {sp_props:+.1f}"
                                )
                            if sp_power is not None:
                                st.markdown(
                                    f"- Power-only spread: {sp_power:+.1f}"
                                )

                            st.markdown(
                                f"- Props used: **{int(fav['props_count'])}**  "
                                f"(avg conf: {fav['avg_confidence']:.1f}%)"
                            )

                            # Team advanced stats summary
                            off = fav.get("OFF_RATING")
                            deff = fav.get("DEF_RATING")
                            net = fav.get("NET_RATING")
                            pace = fav.get("PACE")
                            w = fav.get("W")
                            l = fav.get("L")
                            w_pct = fav.get("W_PCT")

                            st.markdown("**Team metrics (NBA_API)**")
                            if w is not None and l is not None and w_pct is not None:
                                st.markdown(
                                    f"- Record: {int(w)}–{int(l)}  "
                                    f"({w_pct*100:.1f}% win)"
                                )
                            if off is not None and deff is not None and net is not None:
                                st.markdown(
                                    f"- OffRtg/DefRtg/NetRtg: "
                                    f"{off:.1f} / {deff:.1f} / {net:.1f}"
                                )
                            if pace is not None:
                                st.markdown(f"- Pace: {pace:.1f}")

                            # Book odds + edges (if available)
                            book_ml = fav.get("book_ml_odds")
                            book_ml_prob = fav.get("book_ml_prob")
                            ml_edge = fav.get("ml_edge_pct")
                            book_spread = fav.get("book_spread")
                            spread_edge = fav.get("spread_edge_pts")

                            if book_ml is not None and book_ml_prob is not None:
                                st.markdown(
                                    f"- Book ML: {int(book_ml)} "
                                    f"({book_ml_prob*100:.1f}% implied)"
                                )
                                if ml_edge is not None:
                                    st.markdown(
                                        f"- ML edge vs book: "
                                        f"{ml_edge:+.1f} percentage points"
                                    )

                            if book_spread is not None:
                                st.markdown(
                                    f"- Book spread: {book_spread:+.1f}"
                                )
                                if spread_edge is not None:
                                    st.markdown(
                                        f"- Spread edge vs book: "
                                        f"{spread_edge:+.1f} pts (model - book)"
                                    )

                        with top_c2:
                            if dog:
                                st.markdown("**Other side (model view)**")
                                st.markdown(f"Team: **{dog['team']}**")

                                d_wp_final = dog.get("final_win_prob_pct")
                                d_final_odds = dog.get("final_ml_odds")
                                d_wp_props = dog.get("win_prob_pct")
                                d_props_odds = dog.get("ml_odds")
                                d_wp_power = dog.get("power_win_pct")
                                d_power_odds = dog.get("power_ml_odds")

                                if d_wp_final is not None:
                                    st.markdown(
                                        f"- Final win %: **{d_wp_final:.1f}%**  "
                                        f"({d_final_odds})"
                                    )
                                if d_wp_props is not None:
                                    st.markdown(
                                        f"- Props model: {d_wp_props:.1f}%  ({d_props_odds})"
                                    )
                                if d_wp_power is not None:
                                    st.markdown(
                                        f"- Power idx: {d_wp_power:.1f}%  ({d_power_odds})"
                                    )

                                d_sp_final = dog.get("final_model_spread")
                                d_sp_props = dog.get("model_spread")
                                d_sp_power = dog.get("power_spread")

                                if d_sp_final is not None:
                                    st.markdown(
                                        f"- Final model spread: **{d_sp_final:+.1f}**"
                                    )
                                if d_sp_props is not None:
                                    st.markdown(
                                        f"- Props-only spread: {d_sp_props:+.1f}"
                                    )
                                if d_sp_power is not None:
                                    st.markdown(
                                        f"- Power-only spread: {d_sp_power:+.1f}"
                                    )

                                st.markdown(
                                    f"- Props used: **{int(dog['props_count'])}**  "
                                    f"(avg conf: {dog['avg_confidence']:.1f}%)"
                                )

                                # Book odds + edges for the dog (if desired, similar to fav)
                                d_book_ml = dog.get("book_ml_odds")
                                d_book_ml_prob = dog.get("book_ml_prob")
                                d_ml_edge = dog.get("ml_edge_pct")
                                d_book_spread = dog.get("book_spread")
                                d_spread_edge = dog.get("spread_edge_pts")

                                if d_book_ml is not None and d_book_ml_prob is not None:
                                    st.markdown(
                                        f"- Book ML: {int(d_book_ml)} "
                                        f"({d_book_ml_prob*100:.1f}% implied)"
                                    )
                                    if d_ml_edge is not None:
                                        st.markdown(
                                            f"- ML edge vs book: "
                                            f"{d_ml_edge:+.1f} percentage points"
                                        )

                                if d_book_spread is not None:
                                    st.markdown(
                                        f"- Book spread: {d_book_spread:+.1f}"
                                    )
                                    if d_spread_edge is not None:
                                        st.markdown(
                                            f"- Spread edge vs book: "
                                            f"{d_spread_edge:+.1f} pts (model - book)"
                                        )

                            else:
                                st.markdown("_No data for other side of this game._")

                        st.markdown("---")
                        wp_final = fav.get("final_win_prob_pct", 0)
                        sp_final = fav.get("final_model_spread", 0)
                        label = f"{fav_team} {sp_final:+.1f} (≈ {wp_final:.1f}% ML)"
                        st.markdown(f"**Blended model lean:** {label}")
                        st.progress(min(max(wp_final / 100.0, 0.0), 1.0))

                        # Show which props contributed
                        with st.expander("Show props used for favorite"):
                            fav_props = edges_df[
                                (edges_df["team"] == fav_team)
                                & (edges_df["opponent"] == opp_team)
                                & (edges_df["market"].isin(["points", "pra", "fs", "threes"]))
                            ].copy()

                            cols_to_show = [
                                "player",
                                "market",
                                "line",
                                "predicted_score",
                                edge_col,
                                rate_col,
                                "confidence_pct",
                                "bet_side",
                            ]

                            for c in cols_to_show:
                                if c not in fav_props.columns:
                                    fav_props[c] = None

                            st.dataframe(
                                fav_props[cols_to_show]
                                .sort_values("confidence_pct", ascending=False),
                                use_container_width=True,
                            )

                        if dog is not None:
                            with st.expander("Show props used for other team"):
                                dog_props = edges_df[
                                    (edges_df["team"] == dog["team"])
                                    & (edges_df["opponent"] == dog["opponent"])
                                    & (edges_df["market"].isin(["points", "pra", "fs", "threes"]))
                                ].copy()

                                cols_to_show = [
                                    "player",
                                    "market",
                                    "line",
                                    "predicted_score",
                                    edge_col,
                                    rate_col,
                                    "confidence_pct",
                                    "bet_side",
                                ]
                                for c in cols_to_show:
                                    if c not in dog_props.columns:
                                        dog_props[c] = None

                                st.dataframe(
                                    dog_props[cols_to_show]
                                    .sort_values("confidence_pct", ascending=False),
                                    use_container_width=True,
                                )

                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button(
                                "Track favorite ML",
                                key=f"track_ml_{game_key}_{k_suffix}",
                            ):
                                bet = {
                                    "bet_category": "game_ml",
                                    "player": fav_team,
                                    "team": fav_team,
                                    "opponent": opp_team,
                                    "market": "ML",
                                    "line": None,
                                    "bet_side": "ML_favorite",
                                    "bet_odds": fav.get("final_ml_odds"),
                                    "predicted_score": None,
                                    "confidence_pct": fav.get("avg_confidence"),
                                    "game_time": None,
                                    "game_key": game_key,
                                    "model_spread": fav.get("final_model_spread"),
                                    "win_prob_pct": fav.get("final_win_prob_pct"),
                                    "actual_stat": None,
                                    "result": None,
                                }
                                st.session_state["bet_tracker"].append(bet)
                                save_bets_to_disk(st.session_state["bet_tracker"])
                                st.success("Favorite ML bet added to tracker.")

                        with col_btn2:
                            if st.button(
                                "Track spread (model)",
                                key=f"track_spread_{game_key}_{k_suffix}",
                            ):
                                bet = {
                                    "bet_category": "game_spread",
                                    "player": fav_team,
                                    "team": fav_team,
                                    "opponent": opp_team,
                                    "market": "spread",
                                    "line": fav.get("final_model_spread"),
                                    "bet_side": "favorite_spread",
                                    "bet_odds": None,
                                    "predicted_score": None,
                                    "confidence_pct": fav.get("avg_confidence"),
                                    "game_time": None,
                                    "game_key": game_key,
                                    "model_spread": fav.get("final_model_spread"),
                                    "win_prob_pct": fav.get("final_win_prob_pct"),
                                    "actual_stat": None,
                                    "result": None,
                                }
                                st.session_state["bet_tracker"].append(bet)
                                save_bets_to_disk(st.session_state["bet_tracker"])
                                st.success("Model spread bet added to tracker.")

                if mobile_mode:
                    for k in range(len(fav_list)):
                        fav = fav_list[k]
                        render_game_card(fav, f"m_{k}")
                else:
                    for idx in range(0, len(fav_list), 2):
                        row_cols = st.columns(2)
                        for j in range(2):
                            k = idx + j
                            if k >= len(fav_list):
                                break
                            fav = fav_list[k]
                            with row_cols[j]:
                                render_game_card(fav, f"d_{k}")

            st.markdown("#### Blended Model Moneylines vs Book (favorites per game)")

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
                "power_win_pct",
                "power_ml_odds",
                "final_win_prob_pct",
                "final_ml_odds",
                "book_ml_odds",
                "book_ml_prob",
                "ml_edge_pct",
            ]
            for c in ml_cols:
                if c not in favorites.columns:
                    favorites[c] = None
            st.dataframe(
                favorites[ml_cols]
                .sort_values("ml_edge_pct", ascending=False),
                use_container_width=True,
            )

            st.markdown("#### Blended Model Spreads vs Book (favorites per game)")
            spread_cols = [
                "game_key",
                "game_label",
                "team",
                "opponent",
                "props_count",
                "avg_confidence",
                "avg_edge",
                "model_spread",
                "power_spread",
                "final_model_spread",
                "book_spread",
                "spread_edge_pts",
                "final_win_prob_pct",
                "final_ml_odds",
            ]
            for c in spread_cols:
                if c not in favorites.columns:
                    favorites[c] = None
            st.dataframe(
                favorites[spread_cols]
                .sort_values("spread_edge_pts", key=lambda s: s.abs(), ascending=False),
                use_container_width=True,
            )


# =====================================================
# TAB 5: BET TRACKER (HIT/MISS + LEARNING)
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

        if "game_time" in bets_df.columns:
            bets_df["game_time"] = pd.to_datetime(bets_df["game_time"], errors="coerce")

        if "bet_category" not in bets_df.columns:
            bets_df["bet_category"] = "player_prop"

        if "actual_stat" not in bets_df.columns:
            bets_df["actual_stat"] = None
        if "result" not in bets_df.columns:
            bets_df["result"] = None

        # Auto-grade only PLAYER PROP bets
        for idx, b in bets_df.iterrows():
            if b.get("bet_category") != "player_prop":
                continue

            player_name = b.get("player")
            market = b.get("market")
            line = b.get("line")
            side = b.get("bet_side")
            gtime = b.get("game_time")

            actual_stat = None
            result = b.get("result") or "Unknown"

            try:
                target_date = None
                if pd.notna(gtime):
                    try:
                        if not isinstance(gtime, pd.Timestamp):
                            gtime_ts = pd.to_datetime(gtime, errors="coerce")
                        else:
                            gtime_ts = gtime
                        if gtime_ts is not None and not pd.isna(gtime_ts):
                            target_date = gtime_ts.date()
                    except Exception:
                        target_date = None

                pid = get_player_id(player_name)
                glog = get_player_gamelog(pid)

                if glog.empty:
                    result = "No game log"
                else:
                    glog = glog.copy()
                    glog["GAME_DATE_ONLY"] = glog["GAME_DATE"].dt.date

                    game_row = pd.DataFrame()

                    if target_date is not None:
                        game_row = glog[glog["GAME_DATE_ONLY"] == target_date]

                        if game_row.empty:
                            # Fallback: closest game within +/-2 days of target_date
                            diffs = glog["GAME_DATE_ONLY"].apply(
                                lambda d: abs((d - target_date).days)
                            )
                            min_idx = diffs.idxmin()
                            if diffs[min_idx] <= 2:
                                game_row = glog.loc[[min_idx]]
                    else:
                        # No game_time recorded: use most recent game if within last 2 days
                        latest_idx = glog["GAME_DATE"].idxmax()
                        latest_date = glog.loc[latest_idx, "GAME_DATE"].date()
                        if (datetime.today().date() - latest_date).days <= 2:
                            game_row = glog.loc[[latest_idx]]

                    if game_row.empty:
                        if target_date is None:
                            result = "No game_time"
                        else:
                            result = "Not played / not final yet"
                    else:
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

            bets_df.at[idx, "actual_stat"] = actual_stat
            bets_df.at[idx, "result"] = result

        st.session_state["bet_tracker"] = bets_df.to_dict(orient="records")
        save_bets_to_disk(st.session_state["bet_tracker"])

        props_view = bets_df[bets_df["bet_category"] == "player_prop"].copy()
        game_view = bets_df[bets_df["bet_category"].isin(["game_ml", "game_spread"])].copy()

        st.markdown("#### Summary (player props)")

        if props_view.empty:
            st.info("No player prop bets tracked yet.")
        else:
            COMPLETED_STATUSES = {"Hit", "Miss", "N/A", "Error", "No stat for market"}
            completed_mask = props_view["result"].isin(COMPLETED_STATUSES)
            completed_props = props_view[completed_mask]
            total_props = len(props_view)
            completed_count = len(completed_props)
            hits_count = (completed_props["result"] == "Hit").sum()
            misses_count = (completed_props["result"] == "Miss").sum()
            denom = hits_count + misses_count
            hit_rate = (hits_count / denom * 100.0) if denom > 0 else 0.0

            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            col_kpi1.metric("Tracked player props", total_props)
            col_kpi2.metric("Completed (graded)", completed_count)
            col_kpi3.metric("Hit rate (Hit vs Miss)", f"{hit_rate:.1f}%")

        st.markdown("#### Summary (game bets – ML & spreads)")

        if game_view.empty:
            st.info("No game-level ML/spread bets tracked yet.")
        else:
            show_cols_game = [
                "bet_category",
                "game_key",
                "team",
                "opponent",
                "market",
                "line",
                "bet_side",
                "bet_odds",
                "model_spread",
                "win_prob_pct",
                "confidence_pct",
                "game_time",
                "result",
            ]
            for c in show_cols_game:
                if c not in game_view.columns:
                    game_view[c] = None

            if mobile_mode:
                with st.expander("Show game bets table"):
                    st.dataframe(game_view[show_cols_game], use_container_width=True)
            else:
                st.dataframe(game_view[show_cols_game], use_container_width=True)

        st.markdown("### Learning from misses (player props only)")

        if props_view.empty:
            st.info("No player prop bets yet to analyze.")
        else:
            learn_df = props_view[props_view["result"].isin(["Hit", "Miss"])].copy()
            if learn_df.empty:
                st.info("No completed Hit/Miss player props yet to learn from.")
            else:
                market_perf = (
                    learn_df.groupby("market")["result"]
                    .value_counts(normalize=True)
                    .rename("rate")
                    .reset_index()
                    .pivot(index="market", columns="result", values="rate")
                    .fillna(0.0)
                )

                if mobile_mode:
                    with st.expander("Hit/Miss rate by market"):
                        st.dataframe(market_perf, use_container_width=True)
                else:
                    st.markdown("**Hit/Miss rate by market**")
                    st.dataframe(market_perf, use_container_width=True)

                if "Hit" in market_perf.columns:
                    best_market = market_perf["Hit"].idxmax()
                    best_rate = market_perf["Hit"].max()
                    worst_market = market_perf["Hit"].idxmin()
                    worst_rate = market_perf["Hit"].min()
                    st.caption(
                        f"Best market so far: **{best_market}** ({best_rate*100:.1f}% hits).  "
                        f"Toughest market: **{worst_market}** ({worst_rate*100:.1f}% hits)."
                    )

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

                learn_df["conf_bucket"] = learn_df["confidence_pct"].apply(conf_bucket)
                bucket_perf = (
                    learn_df.groupby("conf_bucket")["result"]
                    .value_counts(normalize=True)
                    .rename("rate")
                    .reset_index()
                    .pivot(index="conf_bucket", columns="result", values="rate")
                    .fillna(0.0)
                    .sort_index()
                )

                if mobile_mode:
                    with st.expander("Hit/Miss rate by confidence bucket"):
                        st.dataframe(bucket_perf, use_container_width=True)
                else:
                    st.markdown("**Hit/Miss rate by confidence bucket**")
                    st.dataframe(bucket_perf, use_container_width=True)

                if "Hit" in bucket_perf.columns:
                    best_bucket = bucket_perf["Hit"].idxmax()
                    best_bucket_rate = bucket_perf["Hit"].max()
                    st.caption(
                        f"Your strongest confidence zone so far is **{best_bucket}** "
                        f"({best_bucket_rate*100:.1f}% hits)."
                    )
