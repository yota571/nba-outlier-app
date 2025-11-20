import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NBA Outlier — Cloud Version",
    layout="wide",
)

st.title("🏀 NBA Outlier — Cloud Version")
st.caption("Cloud-Optimized Version • Fast • Stable • No Local Files Needed")


# ============================================================
# PRIZEPICKS WRAPPER (INLINE VERSION)
# ============================================================
class PrizePicks:
    BASE = "https://api.prizepicks.com/projections"

    @staticmethod
    def get_data():
        """Fetch raw PrizePicks projections JSON."""
        try:
            r = requests.get(PrizePicks.BASE, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            st.error(f"❌ Failed loading PrizePicks data: {e}")
            return None

    @staticmethod
    def players_df():
        """Convert PrizePicks projection data into DataFrame."""
        data = PrizePicks.get_data()
        if data is None:
            return pd.DataFrame()

        included = {item["id"]: item for item in data.get("included", [])}

        rows = []
        for p in data["data"]:
            attr = p["attributes"]

            player_id = attr["player_id"]
            player = included.get(str(player_id), {}).get("attributes", {})

            stat_type = included.get(str(attr["stat_type_id"]), {}).get("attributes", {}).get("name")

            rows.append({
                "player": player.get("name", "Unknown"),
                "team": player.get("team", "—"),
                "league": player.get("league", "NBA"),
                "stat_type": stat_type,
                "line": attr.get("line_score"),
                "projection_id": p["id"]
            })

        return pd.DataFrame(rows)


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data(ttl=120)
def load_prizepicks():
    df = PrizePicks.players_df()
    if df.empty:
        st.warning("No NBA PrizePicks data found.")
    return df

df = load_prizepicks()


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("Filters")

stat_filter = st.sidebar.selectbox(
    "Stat Type",
    sorted(df["stat_type"].dropna().unique()) if not df.empty else ["Points", "Rebounds", "Assists"]
)

search_player = st.sidebar.text_input("Search Player")

filtered = df.copy()

if search_player:
    filtered = filtered[filtered["player"].str.contains(search_player, case=False)]

if stat_filter:
    filtered = filtered[filtered["stat_type"] == stat_filter]


# ============================================================
# DISPLAY
# ============================================================
st.subheader(f"📊 NBA PrizePicks — {stat_filter}")

if filtered.empty:
    st.warning("No props match your filters.")
else:
    st.dataframe(filtered, use_container_width=True)


# ============================================================
# OUTLIER LOGIC (Simple Scoring Placeholder)
# ============================================================
st.subheader("🔥 Outlier Detector (Simple Version)")

if not filtered.empty:
    outlier_df = filtered.copy()

    # simple detection placeholder (replace later with ML)
    outlier_df["z_score"] = (outlier_df["line"] - outlier_df["line"].mean()) / outlier_df["line"].std()

    st.dataframe(outlier_df.sort_values("z_score"), use_container_width=True)

else:
    st.info("Load data to see outlier predictions.")

st.caption("NBA Outlier Cloud Engine v2.0 • Ready for machine learning integration.")
