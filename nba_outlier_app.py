def build_moneyline_predictions(
    edges: pd.DataFrame,
    scoring_markets=None,
) -> pd.DataFrame:
    """
    Build heuristic game moneyline predictions using prop edges.

    - Only use scoring-related markets (default: points, pra, fs, threes)
    - Signal per prop = edge * confidence * direction (+ for Over, - for Under)
    - Aggregate per (game_key, team, opponent)
    - Normalize within each game_key so the two teams share the win probability
    """
    if scoring_markets is None:
        scoring_markets = ["points", "pra", "fs", "threes"]

    if edges.empty:
        return pd.DataFrame()

    required_cols = {"team", "opponent", "market", edge_col, "confidence_pct"}
    if not required_cols.issubset(edges.columns):
        return pd.DataFrame()

    # only scoring-related markets
    df_ml = edges.copy()
    df_ml = df_ml[df_ml["market"].isin(scoring_markets)]

    if df_ml.empty:
        return pd.DataFrame()

    # canonical game key so both sides of the matchup are grouped together
    def make_game_key(row):
        t = str(row.get("team", "NA"))
        o = str(row.get("opponent", "NA"))
        teams_sorted = sorted([t, o])
        return " vs ".join(teams_sorted)

    df_ml["game_key"] = df_ml.apply(make_game_key, axis=1)

    # per-prop signal
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

    # aggregate per team in each game_key
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

    # raw win prob from logistic over avg_signal
    def to_prob(sig):
        try:
            return 1.0 / (1.0 + math.exp(-0.8 * sig))
        except OverflowError:
            return 1.0 if sig > 0 else 0.0

    grp["raw_prob"] = grp["avg_signal"].apply(to_prob)

    # normalize within each game_key so both sides share 1.0 total
    grp["sum_prob_game"] = grp.groupby("game_key")["raw_prob"].transform("sum")

    def norm_prob(row):
        s = row["sum_prob_game"]
        rp = row["raw_prob"]
        if s <= 0:
            return 0.5
        p = rp / s
        # keep it away from absolute 0/1
        return max(0.05, min(0.95, p))

    grp["win_prob"] = grp.apply(norm_prob, axis=1)
    grp["ml_odds"] = grp["win_prob"].apply(prob_to_american)
    grp["win_prob_pct"] = (grp["win_prob"] * 100).round(1)

    # recreate a readable game_label (team vs opponent) for display
    grp["game_label"] = grp.apply(
        lambda r: f"{r['team']} vs {r['opponent']}", axis=1
    )

    # order by biggest favorites
    grp = grp.sort_values(["win_prob", "game_key"], ascending=[False, True])

    return grp
