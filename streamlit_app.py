import streamlit as st
import pandas as pd
import json
from io import StringIO

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NBA Outlier — Player Cards View",
    layout="wide",
)

st.title("🏀 NBA Outlier — Player Cards View")
st.caption("PrizePicks-style layout • Player cards • Alt lines per prop • File-upload powered")


# ============================================================
# DATA INGEST HELPERS (FROM UPLOADED FILE)
# ============================================================
def parse_prizepicks_json(pp_json: dict) -> pd.DataFrame:
    """
    Parse a raw PrizePicks projections JSON (same structure as the API)
    into a flat DataFrame with one row per projection.
    """
    included = {item["id"]: item for item in pp_json.get("included", [])}

    rows = []
    for proj in pp_json.get("data", []):
        attr = proj.get("attributes", {})

        player_id = str(attr.get("player_id"))
        stat_type_id = str(attr.get("stat_type_id"))
        game_id = str(attr.get("game_id")) if attr.get("game_id") else None

        player = included.get(player_id, {}).get("attributes", {}) if player_id in included else {}
        stat_type = included.get(stat_type_id, {}).get("attributes", {}) if stat_type_id in included else {}
        game = included.get(game_id, {}).get("attributes", {}) if game_id and game_id in included else {}

        rows.append(
            {
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
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty and "league" in df.columns:
        df = df[df["league"] == "NBA"]

    return df


def load_from_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Accept a CSV or JSON file and return a normalized projections DataFrame.
    - JSON: must be raw PrizePicks API structure (data + included).
    - CSV: must already contain columns like player, team, stat_type, line, etc.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".json"):
            # Parse as PrizePicks JSON
            pp_json = json.load(uploaded_file)
            return parse_prizepicks_json(pp_json)

        elif name.endswith(".csv"):
            # Assume it's already a flat props CSV
            df = pd.read_csv(uploaded_file)

            # Try to normalize a bit
            # We expect at least: player, team, stat_type, line
            expected_cols = {"player", "team", "stat_type", "line"}
            if not expected_cols.issubset(df.columns):
                st.warning(
                    "Uploaded CSV does not contain the expected columns: "
                    "`player`, `team`, `stat_type`, `line`. "
                    "The app will still try to run, but some views may be empty."
                )

            # Add missing columns with defaults
            for col in [
                "league",
                "position",
                "stat_type_abbr",
                "odds_type",
                "home_team",
                "away_team",
                "game_started",
                "projection_id",
            ]:
                if col not in df.columns:
                    df[col] = None

            # Default league to NBA if missing
            if "league" in df.columns and df["league"].isna().all():
                df["league"] = "NBA"

            return df

        else:
            st.error("Unsupported file type. Please upload a JSON or CSV file.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return pd.DataFrame()


# ============================================================
# FILE UPLOADER
# ============================================================
st.sidebar.markdown("### 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload PrizePicks JSON or props CSV",
    type=["json", "csv"],
    help=(
        "• JSON: raw response from PrizePicks projections API (data + included)\n"
        "• CSV: your own export with at least columns: player, team, stat_type, line"
    ),
)

df = load_from_uploaded_file(uploaded_file)

if df.empty:
    st.warning(
        "No data loaded yet. Upload a PrizePicks JSON or CSV file in the sidebar "
        "to populate the player cards."
    )

if "selected_player" not in st.session_state:
    st.session_state.selected_player = None
if "selected_stat" not in st.session_state:
    st.session_state.selected_stat = None


# ============================================================
# SIDEBAR FILTERS (only when data exists)
# ============================================================
st.sidebar.markdown("---")
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

# Apply filters
filtered = df.copy()

if not filtered.empty:
    if selected_stat_filter:
        filtered = filtered[filtered["stat_type"].isin(selected_stat_filter)]

    if search_player:
        filtered = filtered[filtered["player"].str.contains(search_player, case=False)]

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
    st.info("No props match your filters right now (or no file uploaded yet).")
else:
    CARDS_PER_ROW = 3

    # Placeholder avatar; can later swap to real headshots
    placeholder_avatar = (
        "https://ui-avatars.com/api/?name={name}&background=random&size=128"
    )

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

                avatar_url = placeholder_avatar.format(
                    name=str(player_name).replace(" ", "+")
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

        # Columns to show if they exist
        possible_cols = [
            "line",
            "odds_type",
            "home_team",
            "away_team",
            "game_started",
        ]
        show_cols = [c for c in possible_cols if c in detail_df.columns]

        st.dataframe(detail_df[show_cols], use_container_width=True)
else:
    st.info(
        "Select a player card and stat to view all alt lines, similar to PrizePicks."
    )


# ============================================================
# FOOTER
# ============================================================
st.caption(
    "NBA Outlier Cloud Engine • Player Card Layout • Uses uploaded PrizePicks data "
    "instead of live API calls (no 403, fully Streamlit Cloud compatible)."
)
