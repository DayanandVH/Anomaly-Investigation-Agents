"""
Baseline anomaly detector.

Deliberately simple (IsolationForest) — the point of this project is the
agent layer on top (RAG + LangGraph reasoning), not novel detection research.
The one thing that matters here: every flagged row comes with WHICH features
look off and by how much, since that's what the agent will explain later.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyFlag:
    """A single flagged (anomalous) row, packaged with WHY it was flagged.

    @dataclass auto-generates __init__, __repr__, etc. from the fields below,
    so we don't have to hand-write that boilerplate.

    Note on `field(default_factory=list)`: for mutable defaults (lists, dicts)
    you can't just write `top_features: List[Dict] = []` — Python would share
    ONE list across every instance of this class, which is a classic bug.
    `default_factory=list` tells it to call list() fresh for each new object.
    """
    row_index: int                          # which row in the original dataframe this is
    anomaly_score: float                    # higher = more anomalous
    top_features: List[Dict[str, Any]] = field(default_factory=list)  # e.g. [{"feature": "V9", "z_score": 5.28, "value": 3.67}]
    raw_row: Dict[str, Any] = field(default_factory=dict)             # the full original row, for reference


class AnomalyDetector:
    """Wraps an IsolationForest with standardized scoring and per-feature
    explanations, so downstream code (and later, the agent) gets structured
    reasons instead of a bare anomaly/normal label.
    """

    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        """
        Args:
            contamination: expected fraction of anomalies in the data (e.g. 0.02 = 2%).
                            IsolationForest uses this to decide its decision threshold.
            random_state: fixed seed so results are reproducible run to run.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.scaler = StandardScaler()   # standardizes features to mean=0, std=1 (needed so z-scores are meaningful)
        self.feature_cols: List[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "AnomalyDetector":
        """Train the detector on the given dataframe's feature columns.

        Fitting the scaler here (not in score()) matters: we want new data scored
        against the mean/std learned from training data, not re-standardized on
        its own — otherwise scores wouldn't be comparable across calls.
        """
        self.feature_cols = feature_cols
        X = self.scaler.fit_transform(df[feature_cols])
        self.model.fit(X)
        return self  # returning self lets you chain: AnomalyDetector().fit(df, cols)

    def score(self, df: pd.DataFrame, top_k: int = 3) -> List[AnomalyFlag]:
        """Score every row in df; return an AnomalyFlag for each row the model
        predicts as anomalous, including the top_k features most responsible.

        Args:
            df: data to score (must contain self.feature_cols)
            top_k: how many top contributing features to report per flagged row
        """
        X = self.scaler.transform(df[self.feature_cols])   # standardize using the TRAINED scaler
        preds = self.model.predict(X)            # -1 = anomaly, 1 = normal
        scores = -self.model.score_samples(X)    # sklearn's raw score is "higher = more normal"; we flip it so higher = more anomalous (more intuitive)

        # Because X is already standardized (mean=0, std=1), each value IS its z-score —
        # i.e. "how many standard deviations from normal is this feature for this row".
        z_scores = X

        flags = []
        for i, (pred, score) in enumerate(zip(preds, scores)):
            if pred != -1:
                continue  # skip rows the model considers normal

            row_z = z_scores[i]
            # argsort by absolute z-score, descending, so we get the features
            # that deviate the MOST (in either direction) first
            top_idx = np.argsort(-np.abs(row_z))[:top_k]

            top_features = []
            for j in top_idx:
                top_features.append({
                    "feature": self.feature_cols[j],
                    "z_score": round(float(row_z[j]), 2),
                    "value": round(float(df[self.feature_cols[j]].iloc[i]), 2),
                })

            flags.append(
                AnomalyFlag(
                    row_index=int(df.index[i]),
                    anomaly_score=round(float(score), 4),
                    top_features=top_features,
                    raw_row=df.iloc[i].to_dict(),
                )
            )
        return flags


if __name__ == "__main__":
    # Quick manual test: load data, fit the detector, print a few flagged rows.
    from data import load_data

    df, source = load_data()
    feature_cols = [c for c in df.columns if c not in ("Class",)]  # everything except the label

    detector = AnomalyDetector(contamination=0.02).fit(df, feature_cols)
    flags = detector.score(df)

    print(f"Data source: {source}")
    print(f"Flagged {len(flags)} / {len(df)} rows as anomalous\n")

    for flag in flags[:3]:
        print(f"Row {flag.row_index} | anomaly_score={flag.anomaly_score}")
        for f in flag.top_features:
            print(f"   {f['feature']}: value={f['value']}, z={f['z_score']}")
        print()
