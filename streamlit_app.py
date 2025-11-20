import requests
import pandas as pd
import streamlit as st
from datetime import datetime

PRIZEPICKS_URL = "https://partner-api.prizepicks.com/projections"

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="NBA Outlier (No Alt Lines)",
    layout="wide",
)

st.title("NBA Outlier – Main Lines Only (v1.9.1-style)")


# =====================================================
# DATA FETCH / TRANSFORM
# =====================================================
def fetch_nba_board():
    """
    Fetch NBA projections from PrizePicks.
    This matches the JSON structure in the file you uploaded.
    """
    params = {
        "league_id": 7,     # NBA
        "per_page": 1000,
        "single_stat": True,
    }
    resp = requests.get(PRIZEPICKS_URL, params=params)
    resp.raise_for_status()
    return resp.json()


def build_player_lookup(included):
    """Map new_player.id -> dict(name, team, position) from `included`."""
    players = {}
    for item in included:
        if item.get("type") == "new_player":
            pid = str(item["id"])
            attrs = item.get("attributes", {})
            players[pid] = {
                "player_name": attrs.get("name"),
                "team": attrs.get("team"),
                "position": attrs.get("position"),
            }
    return players


def projections_to_df(raw):
    """
    Turn PrizePicks JSON into a flat DataFrame.
    Only uses fields we actually need.
    """
    data = raw.get("data", [])
    included = raw.get("included", [])
    player_map = build_player_lookup(included)

    rows = []
    for proj in data:
        attrs = proj.get("attributes", {})
        rel = proj.get("relationships", {})
        new_player_rel = rel.get("new_player", {}).get("data")

        player_id = None
        if new_player_rel:
            player_id = str(new_player_rel.get("id"))

        player_info = player_map.get(player_id, {}) if player_id else {}

        rows.append(
            {
                "projection_id": proj.get("id"),
                "player_id": player_id,
                "player_name": player_info.get("player_name"),
                "team": player_info.get("team"),
                "position": player_info.get("position"),
                "description": attrs.get("description"),
                "event_type": attrs.get("event_type"),           # team / combo / etc.
                "stat_display_name": attrs.get("stat_display_name"),
                "stat_type": attrs.get("stat_type"),
                "line_score": attrs.get("line_score"),
                "odds_type": attrs.get("odds_type"),
                "adjusted_odds": attrs.get("adjusted_odds"),
                "start_time": attrs.get("start_time"),
                "today": attrs.get("today"),
            }
        )

    df = pd.DataFrame(rows)
    return df


def filter_main_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Emulate 'pre-alt-lines' behavior:
    - Keep only odds_type == 'standard'
    - Drop adjusted/boosted lines (adjusted_odds == True)
    """
    if df.empty:
        return df

    if "odds_type" not in df.columns:
        return df

    mask_standard = df["odds_type"].eq("standard")

    if "adjusted_odds" in df.columns:
        mask_not_adjusted = ~df["adjusted_odds"].fillna(False)
    else:
        mask_not_adjusted = True

    return df[mask_standard & mask_not_adjusted]


def compute_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stub for your ML / edge logic.

    Right now this just creates an 'edge' column = 0.
    Drop your model predictions here and compute:
        df['model_line'] = ...
        df['edge'] = df['model_line'] - df['line_score']
    """
    df = df.copy()
    df["edge"] = 0.0
    return df


# =====================================================
# MAIN APP FLOW
# =====================================================
@st.cache_data(show_spinner=True)
def load_board():
    raw = fetch_nba_board()
    df = projections_to_df(raw)
    df_main = filter_main_lines(df)  # <-- key change: remove alt lines
    df_with_edges = compute_edges(df_main)
    return df_with_edges


with st.spinner("Loading NBA board..."):
    try:
        df_board = load_board()
    except Exception as e:
        st.error(f"Error loading board: {e}")
        st.stop()

if df_board.empty:
    st.warning("No projections found.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

teams = sorted([t for t in df_board["team"].dropna().unique()])
stats = sorted([s for s in df_board["stat_display_name"].dropna().unique()])

selected_team = st.sidebar.multiselect("Team", teams, default=teams)
selected_stat = st.sidebar.multiselect("Stat Type", stats, default=stats)

min_edge = st.sidebar.number_input("Min Edge (absolute)", min_value=0.0, value=0.0, step=0.5)

df_filtered = df_board[
    df_board["team"].isin(selected_team)
    & df_board["stat_display_name"].isin(selected_stat)
]

# Sort by edge desc (largest positive/negative)
df_filtered["abs_edge"] = df_filtered["edge"].abs()
df_filtered = df_filtered.sort_values("abs_edge", ascending=False)

if min_edge > 0:
    df_filtered = df_filtered[df_filtered["abs_edge"] >= min_edge]

st.subheader("Main-line Projections (Alt Lines Removed)")

st.dataframe(
    df_filtered[
        [
            "player_name",
            "team",
            "position",
            "stat_display_name",
            "stat_type",
            "line_score",
            "odds_type",
            "edge",
        ]
    ],
    use_container_width=True,
)
