"""Representative-day selection for the capacity-expansion LP.

Solving dispatch for every hour of every weather year is unnecessary for a
planning model. We cluster all weather-year days on their (load, solar)
daily profiles with k-means, use the real day nearest each centroid (a
medoid, so the hours stay physically consistent), and weight it by cluster
size. The highest-load day is always forced in with a small weight so the
LP sees the extreme it has to survive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def select(ens: pd.DataFrame, k: int = 16, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (load[d, 24], solar_cf[d, 24], weight_days_per_year[d])."""
    n_years = ens["weather_year"].nunique()
    load = ens["load_mw"].to_numpy().reshape(-1, 24)
    solar = ens["solar_cf"].to_numpy().reshape(-1, 24)
    feats = np.hstack([load / load.max(), solar])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(feats)

    medoids, weights = [], []
    for c in range(k):
        members = np.flatnonzero(km.labels_ == c)
        dist = np.linalg.norm(feats[members] - km.cluster_centers_[c], axis=1)
        medoids.append(members[np.argmin(dist)])
        weights.append(len(members) / n_years)

    peak_day = int(np.argmax(load.max(axis=1)))
    if peak_day not in medoids:
        c = km.labels_[peak_day]
        shift = 1.0 / n_years
        weights[c] -= shift
        medoids.append(peak_day)
        weights.append(shift)

    idx = np.array(medoids)
    return load[idx], solar[idx], np.array(weights)
