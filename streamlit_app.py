import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NBA Outlier — Player Cards View",
    layout="wide",
)

st.title("🏀 NBA Outlier — Player Cards View")
st.caption("PrizePicks-style layout • Player cards • Alt lines per prop • Static JSON data")


# ============================================================
# DATA PARSING FROM LOCAL prizepicks.json
# ============================================================

DATA_PATH = Path(__file__).parent / "prizepicks.json"


def build_df_from_raw(raw: dict) -> pd.DataFrame:
    """
    Convert a PrizePicks projections JSON (data + included) into a flat DataFrame.
    This matches the structure in your uploaded prizepicks.json.
    """

    # Index included by type + id for quick lookup
    by_type = {}
    for item in raw.get("included", []):
        t = item.get("type")
        i = str(item.get("id"))
        by_type.setdefault(t, {})[i] = item.get("attributes", {})

    players = by_type.get("new_player", {})
    games = by_type.get("game", {})

    rows = []
    for proj in raw.get("data", []):
        if proj.get("type") != "projection":
            continue

        attr = proj.get("attributes", {})
        rel = proj.get("relationships", {})

        # ----- Player -----
        new_player_id = (
            rel.get("new_player", {})
            .get("data", {})
            .get("id")
        )
        player_attr = players.get(str(new_player_id), {}) if new_player_id else {}

        # ----- Game / teams -----
        game_id = (
            rel.get("game", {})
            .get("data", {})
            .get("id")
        )
        game_attr = games.get(str(game_id), {}) if game_id else {}
        metadata = game_attr.get("metadata", {}) or {}
        game_info = metadata.get("game_info", {}) or {}
        teams = game_info.get("teams", {}) or {}
        away = teams.get("away", {}) or {}
        home = teams.get("home", {}) or {}

        rows.append(
            {
                "projection_id": proj.get("id"),
                "player": player_attr.get("name")
                or player_attr.get("display_name")
                or "Unknown",
                "display_name": player_attr.get("display_name"),
                "team": player_attr.get("team"),
                "team_name": player_attr.get("team_name"),
                "league": player_attr.get("league"),
                "position": player_attr.get("position"),
                "image_url": player_attr.get("image_url"),
                "stat_type": attr.get("stat_type"),  # e.g. "Receiving Yards"
                "stat_display_name": attr.get("stat_display_name"),
                "line": attr.get("line_score"),
                "odds_type": attr.get("odds_type"),
                "projection_type": attr.get("projection_type"),
                "start_time": attr.get("start_time"),
                "status": attr.get("status"),
                "home_team": home.get("abbreviation"),
                "away_team": away.get("abbreviation"),
            }
        )

    df = pd.DataFrame(rows)

    # Optional: if you only want a certain league, filter here
    # e.g., df = df[df["league"] == "NBA"]
    return df


@st.cache_data(ttl=0)
def load_projections() -> pd.DataFrame:
    """Load projections from local prizepicks.json in the repo."""
    if not DATA_PATH.exists():
        st.error(
            "prizepicks.json was not found next to app.py.\n\n"
            "Add your PrizePicks JSON file to the repo root (same folder as app.py) "
            "and redeploy."
        )
        return pd.DataFrame()

    try:
        with DATA_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        st.error(f"Error reading prizepicks.json: {e}")
        return pd.DataFrame()

    df = build_df_from_raw(raw)

    if df.empty:
        st.warning("prizepicks.json loaded, but no projections were parsed.")
    return df


df = load_projections()

# Keep selection across reruns
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None
if "selected_stat" not in st.session_state:
    st.session_state.selected_stat = None


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("Filters")

if not df.empty and "stat_type" in df.columns:
    stat_options = sorted(df["stat_type"].dropna().unique())
else:
    stat_options = []

selected_stat_filter = st.sidebar.multiselect(
    "Stat Types",
    options=stat_options,
    default=stat_options[:5] if len(stat_options) > 5 else stat_options,
)

search_player = st.sidebar.text_input("Search Player")

filtered = df.copy()

if not filtered.empty:
    if selected_stat_filter:
        filtered = filtered[filtered["stat_type"].isin(selected_stat_filter)]

    if search_player:
        filtered = filtered[
            filtered["player"].str.contains(search_player, case=False, na=False)
        ]

# Build unique player+stat combos for cards
if not filtered.empty:
    card_df = (
        filtered.groupby(["player", "team", "stat_type"], as_index=False)
        .agg(
            lines_count=("line", "count"),
            min_line=("line", "min"),
            max_line=("line", "max"),
        )
    )
else:
    card_df = pd.DataFrame(
        columns=["player", "team", "stat_type", "lines_count", "min_line", "max_line"]
    )


# ============================================================
# PLAYER CARD GRID
# ============================================================
st.subheader("📇 Player Prop Cards")

if card_df.empty:
    if df.empty:
        st.info(
            "No data available. Make sure prizepicks.json is in the app folder "
            "and has the PrizePicks projections structure."
        )
    else:
        st.info("No props match your filters right now.")
else:
    CARDS_PER_ROW = 3

    for i in range(0, len(card_df), CARDS_PER_ROW):
        row_slice = card_df.iloc[i : i + CARDS_PER_ROW]
        cols = st.columns(len(row_slice))

        for idx, (_, row) in enumerate(row_slice.iterrows()):
            with cols[idx]:
                player_name = row["player"]
                team = row["team"]
                stat_type = row["stat_type"]
                lines_count = int(row["lines_count"])
                min_line = row["min_line"]
                max_line = row["max_line"]

                # Try to pull the player's image from the full df
                subset = df[(df["player"] == player_name) & (df["team"] == team)]
                image_url = subset["image_url"].dropna().iloc[0] if not subset.empty else None

                if image_url:
                    st.image(image_url, width=96)
                else:
                    # Fallback avatar if PrizePicks has no image_url
                    avatar_url = (
                        "https://ui-avatars.com/api/?name="
                        + str(player_name).replace(" ", "+")
                        + "&background=random&size=128"
                    )
                    st.image(avatar_url, width=96)

                st.markdown(f"**{player_name}**")
                st.markdown(f"🧢 Team: `{team}`")
                st.markdown(f"📊 Stat: **{stat_type}**")
                st.markdown(f"🔢 Alt lines: `{lines_count}`")
                if pd.notna(min_line) and pd.notna(max_line):
                    st.markdown(f"Range: `{min_line}` — `{max_line}`")

                if st.button("View Lines", key=f"view_{player_name}_{stat_type}"):
                    st.session_state.selected_player = player_name
                    st.session_state.selected_stat = stat_type


# ============================================================
# DETAIL PANEL: ALT LINES FOR SELECTED PLAYER + STAT
# ============================================================
st.markdown("---")
st.subheader("🎯 Selected Prop Alt Lines")

if st.session_state.selected_player and st.session_state.selected_stat:
    sel_player = st.session_state.selected_player
    sel_stat = st.session_state.selected_stat

    st.markdown(f"Showing all alt lines for **{sel_player} — {sel_stat}**")

    detail_df = filtered[
        (filtered["player"] == sel_player) & (filtered["stat_type"] == sel_stat)
    ].copy()

    if detail_df.empty:
        st.warning("No lines found for this player/stat (maybe filters changed).")
    else:
        detail_df = detail_df.sort_values("line")

        cols_to_show = [
            "line",
            "stat_display_name",
            "odds_type",
            "home_team",
            "away_team",
            "start_time",
            "status",
        ]
        cols_to_show = [c for c in cols_to_show if c in detail_df.columns]

        st.dataframe(detail_df[cols_to_show], use_container_width=True)
else:
    st.info("Select a player card and stat to view all alt lines, similar to PrizePicks.")


# ============================================================
# FOOTER
# ============================================================
st.caption(
    "NBA Outlier Cloud Engine • Player Card Layout • Reads local prizepicks.json "
    "instead of calling the live API (no 403, fully Streamlit Cloud compatible)."
)
