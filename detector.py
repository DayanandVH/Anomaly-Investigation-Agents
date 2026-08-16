"""
Baseline anomaly detector.

Deliberately simple (IsolationForest) — the point of this project is the
agent layer on top (RAG + LangGraph reasoning), not novel detection research.
The one thing that matters here: every flagged row comes with WHICH features
look off and by how much, since that's what the agent will explain later.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyFlag:
    row_index: int
    anomaly_score: float          # higher = more anomalous
    top_features: list[dict] = field(default_factory=list)  # [{"feature": str, "z_score": float, "value": float}]
    raw_row: dict = field(default_factory=dict)


class AnomalyDetector:
    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "AnomalyDetector":
        self.feature_cols = feature_cols
        X = self.scaler.fit_transform(df[feature_cols])
        self.model.fit(X)
        return self

    def score(self, df: pd.DataFrame, top_k: int = 3) -> list[AnomalyFlag]:
        """Score every row; return AnomalyFlag objects for rows predicted anomalous,
        each carrying the top_k features most responsible (by |z-score|)."""
        X = self.scaler.transform(df[self.feature_cols])
        preds = self.model.predict(X)          # -1 = anomaly, 1 = normal
        scores = -self.model.score_samples(X)   # flip sign: higher = more anomalous

        z_scores = X  # already standardized by the scaler, so this *is* the z-score

        flags = []
        for i, (pred, score) in enumerate(zip(preds, scores)):
            if pred != -1:
                continue
            row_z = z_scores[i]
            top_idx = np.argsort(-np.abs(row_z))[:top_k]
            top_features = [
                {
                    "feature": self.feature_cols[j],
                    "z_score": round(float(row_z[j]), 2),
                    "value": round(float(df[self.feature_cols[j]].iloc[i]), 2),
                }
                for j in top_idx
            ]
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
    from data import load_data

    df, source = load_data()
    feature_cols = [c for c in df.columns if c not in ("Class",)]

    detector = AnomalyDetector(contamination=0.02).fit(df, feature_cols)
    flags = detector.score(df)

    print(f"Data source: {source}")
    print(f"Flagged {len(flags)} / {len(df)} rows as anomalous\n")

    for flag in flags[:3]:
        print(f"Row {flag.row_index} | anomaly_score={flag.anomaly_score}")
        for f in flag.top_features:
            print(f"   {f['feature']}: value={f['value']}, z={f['z_score']}")
        print()
