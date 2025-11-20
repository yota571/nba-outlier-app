"""
train_model.py

Trains a classification model P(Over) using the features created in
build_training_data.py and saves it as over_model.pkl.

The feature columns line up with ML_FEATURE_COLS used in the app.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

DATA_DIR = "data"
IN_FILE = os.path.join(DATA_DIR, "training_data.csv")
MODEL_FILE = "over_model.pkl"

FEATURE_COLS = [
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


def main():
    if not os.path.exists(IN_FILE):
        raise FileNotFoundError(
            f"{IN_FILE} not found. Run build_training_data.py first."
        )

    df = pd.read_csv(IN_FILE)

    # Drop any rows with missing feature or label
    df = df.dropna(subset=FEATURE_COLS + ["y_over"])

    X = df[FEATURE_COLS]
    y = df["y_over"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")

    joblib.dump(model, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
