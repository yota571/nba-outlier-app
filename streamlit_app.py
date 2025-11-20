import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NBA Outlier — Player Cards",
    layout="wide",
)

st.title("🏀 NBA Outlier — Player Cards View")
st.caption(
    "PrizePicks-style layout • Player cards • Alt lines per prop • "
    "Reads local prizepicks.json (no live API calls)."
)

# ============================================================
# LOAD LOCAL PRIZEPICKS.JSON (SAME FOLDER AS THIS FILE)
# ============================================================

DATA_PATH = Path(__file__).parent / "prizepicks.json"


def build_df_from_raw(raw: dict) -> pd.DataFrame:
    """
    Convert a PrizePicks projections JSON (data + included)
    into a flat DataFrame with one row per projection.

    Handles both:
    - older style: attributes.player_id, attributes.game_id, included indexed by id
    - newer style: relationships.new_player, relationships.game, included typed
    """

    included = raw.get("included", []) or []

    # Index included by id (string) for flexible lookup
    included_by_id = {str(item.get("id")): item for item in included}

    rows = []
    for proj in raw.get("data", []):
        if proj.get("type") != "projection":
            continue

        attr = proj.get("attributes", {}) or {}
        rel = proj.get("relationships", {}) or {}

        # ---------------- Player lookup ----------------
        player_attr = {}
        player_id = attr.get("player_id")

        # Older style: attributes.player_id
        if player_id is not None:
            player_item = included_by_id.get(str(player_id), {})
            player_attr = player_item.get("attributes", {}) or {}

        # Newer style: relationships.new_player.data.id
        if not player_attr and "new_player" in rel:
            np_rel = rel["new_player"].get("data") or {}
            np_id = np_rel.get("id")
            if np_id is not None:
                player_item = included_by_id.get(str(np_id), {})
                player_attr = player_item.get("attributes", {}) or {}

        # ---------------- Game lookup ----------------
        game_attr = {}
        game_id = attr.get("game_id")

        # Older style: attributes.game_id
        if game_id is not None:
            game_item = included_by_id.get(str(game_id), {})
            game_attr = game_item.get("attributes", {}) or {}

        # Newer style: relationships.game.data.id
        if not game_attr and "game" in rel:
            g_rel = rel["game"].get("data") or {}
            g_id = g_rel.get("id")
            if g_id is not None:
                game_item = included_by_id.get(str(g_id), {})
                game_attr = game_item.get("attributes", {}) or {}

        # Game fields (best-effort across formats)
        game_started = game_attr.get("started_at") or attr.get("start_time")
        home_team = game_attr.get("home_team")
        away_team = game_attr.get("away_team")

        rows.append(
            {
                "projection_id": proj.get("id"),
                "player": player_attr.get("name")
                or player_attr.get("display_name")
                or "Unknown",
                "team": player_attr.get("team"),
                "league": player_attr.get("league", "NBA"),
                "position": player_attr.get("position"),
                "image_url": player_attr.get("image_url"),
                "stat_type": attr.get("stat_type"),
                "stat_type_abbr": attr.get("stat_type_abbr")
                or attr.get("stat_display_name"),
                "line": attr.get("line_score"),
                "odds_type": attr.get("odds_type"),
                "game_started": game_started,
                "home_team": home_team,
                "away_team": away_team,
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty and "league" in df.columns:
        df = df[df["league"] == "NBA"]

    return df


@st.cache_data(ttl=0)
def load_projections() -> pd.DataFrame:
    """Load projections from prizepicks.json next to this file."""
    if not DATA_PATH.exists():
        st.error(
            "prizepicks.json was not found next to streamlit_app.py.\n\n"
            "Add your PrizePicks JSON file to this repo (same folder) and redeploy."
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
            "No data available. Make sure prizepicks.json is in the repo "
            "next to streamlit_app.py and that it has PrizePicks projections."
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
                image_url = (
                    subset["image_url"].dropna().iloc[0] if not subset.empty else None
                )

                if image_url:
                    st.image(image_url, width=96)
                else:
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
            "stat_type_abbr",
            "odds_type",
            "home_team",
            "away_team",
            "game_started",
        ]
        cols_to_show = [c for c in cols_to_show if c in detail_df.columns]

        st.dataframe(detail_df[cols_to_show], use_container_width=True)
else:
    st.info("Select a player card and stat to view all alt lines, similar to PrizePicks.")

# ============================================================
# FOOTER
# ============================================================
st.caption(
    "NBA Outlier Cloud Engine • Player Card Layout • Uses local prizepicks.json "
    "(no live PrizePicks API calls, so no 403 issues on Streamlit Cloud)."
)
