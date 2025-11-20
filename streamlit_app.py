import streamlit as st
import pandas as pd
import requests

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NBA Outlier — Player Cards View",
    layout="wide",
)

st.title("🏀 NBA Outlier — Player Cards View")
st.caption("PrizePicks-style layout • Player cards • Alt lines per prop")


# ============================================================
# PRIZEPICKS WRAPPER (WITH 403 FIX)
# ============================================================
class PrizePicks:
    BASE = "https://api.prizepicks.com/projections"

    @staticmethod
    def get_data():
        """Fetch PrizePicks projections using spoofed browser headers (fixes 403)."""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://www.prizepicks.com",
                "Referer": "https://www.prizepicks.com/",
            }

            params = {
                "league_id": 7,  # NBA
            }

            r = requests.get(
                PrizePicks.BASE,
                params=params,
                headers=headers,
                timeout=10,
            )
            r.raise_for_status()
            return r.json()

        except Exception as e:
            st.error(f"❌ Failed loading PrizePicks data: {e}")
            return None

    @staticmethod
    def projections_df():
        """
        Build a detailed DataFrame:
        - One row per projection (includes alt lines)
        - Keeps info about player, team, stat_type, line, and game.
        """
        data = PrizePicks.get_data()
        if data is None:
            return pd.DataFrame()

        included = {item["id"]: item for item in data.get("included", [])}

        rows = []
        for proj in data.get("data", []):
            attr = proj.get("attributes", {})

            player_id = str(attr.get("player_id"))
            stat_type_id = str(attr.get("stat_type_id"))
            game_id = str(attr.get("game_id")) if attr.get("game_id") else None

            player = included.get(player_id, {}).get("attributes", {}) if player_id in included else {}
            stat_type = included.get(stat_type_id, {}).get("attributes", {}) if stat_type_id in included else {}
            game = included.get(game_id, {}).get("attributes", {}) if game_id and game_id in included else {}

            rows.append({
                "projection_id": proj.get("id"),
                "player": player.get("name", "Unknown"),
                "team": player.get("team", "—"),
                "league": player.get("league", "NBA"),
                "position": player.get("position", ""),
                "stat_type": stat_type.get("name"),
                "stat_type_abbr": stat_type.get("abbr"),
                "line": attr.get("line_score"),
                "odds_type": attr.get("odds_type"),
                "game_started": game.get("started_at"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
            })

        df = pd.DataFrame(rows)

        # Filter to NBA only, just in case
        if not df.empty:
            df = df[df["league"] == "NBA"]

        return df


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=120)
def load_projections():
    df = PrizePicks.projections_df()
    if df.empty:
        st.warning("No NBA PrizePicks projections available at the moment.")
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

if not df.empty:
    stat_options = sorted(df["stat_type"].dropna().unique())
else:
    stat_options = []

selected_stat_filter = st.sidebar.multiselect(
    "Stat Types",
    options=stat_options,
    default=stat_options[:5] if len(stat_options) > 5 else stat_options,
)

search_player = st.sidebar.text_input("Search Player")

# Apply filters
filtered = df.copy()

if selected_stat_filter:
    filtered = filtered[filtered["stat_type"].isin(selected_stat_filter)]

if search_player:
    filtered = filtered[filtered["player"].str.contains(search_player, case=False)]

# Build unique player+stat combos for cards
if not filtered.empty:
    card_df = (
        filtered
        .groupby(["player", "team", "stat_type"], as_index=False)
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
    st.info("No props match your filters right now.")
else:
    CARDS_PER_ROW = 3

    # Placeholder avatar; can later swap to real headshots
    placeholder_avatar = "https://ui-avatars.com/api/?name={name}&background=random&size=128"

    for i in range(0, len(card_df), CARDS_PER_ROW):
        row_slice = card_df.iloc[i:i + CARDS_PER_ROW]
        cols = st.columns(len(row_slice))

        for idx, (_, row) in enumerate(row_slice.iterrows()):
            with cols[idx]:
                player_name = row["player"]
                team = row["team"]
                stat_type = row["stat_type"]
                lines_count = int(row["lines_count"])
                min_line = row["min_line"]
                max_line = row["max_line"]

                avatar_url = placeholder_avatar.format(name=player_name.replace(" ", "+"))
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
        (filtered["player"] == sel_player) &
        (filtered["stat_type"] == sel_stat)
    ].copy()

    if detail_df.empty:
        st.warning("No lines found for this player/stat (maybe filters changed).")
    else:
        detail_df = detail_df.sort_values("line")

        show_cols = [
            "line",
            "odds_type",
            "home_team",
            "away_team",
            "game_started",
        ]

        st.dataframe(detail_df[show_cols], use_container_width=True)
else:
    st.info("Select a player card and stat to view all alt lines, similar to PrizePicks.")


# ============================================================
# FOOTER
# ============================================================
st.caption("NBA Outlier Cloud Engine • Player Card Layout • 403-fixed PrizePicks integration • Ready for ML.")
