"""Reproducible Elo tournament ratings (iohinspector version is unseeded)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
from skelo.model.elo import EloEstimator


def get_tournament_ratings(
    data: pl.DataFrame,
    alg_vars: Iterable[str] = ("algorithm_name",),
    fid_vars: Iterable[str] = ("function_name",),
    fval_var: str = "raw_y",
    nrounds: int = 25,
    maximization: bool = False,
    return_as_pandas: bool = True,
    random_state: int | None = 42,
) -> pl.DataFrame | pd.DataFrame:
    """Same logic as ``iohinspector.metrics.ranking.get_tournament_ratings``,
    but with a fixed RNG and shuffle so results are reproducible for a given
    ``random_state``.

    Pass ``random_state=None`` to match upstream stochastic behaviour (not
    recommended for published tables).
    """
    fids = data[fid_vars].unique()
    aligned_comps = data.pivot(
        index=alg_vars,
        on=fid_vars,
        values=fval_var,
        aggregate_function=pl.element(),
    )
    players = aligned_comps[alg_vars]
    n_players = players.shape[0]
    comp_arr = np.array(aligned_comps[aligned_comps.columns[len(alg_vars) :]])

    rng = (
        np.random.default_rng()
        if random_state is None
        else np.random.default_rng(random_state)
    )
    fids_idx = [i for i in range(len(fids))]
    lplayers = [i for i in range(n_players)]
    records = []
    for r in range(nrounds):
        for fid in fids_idx:
            for p1 in lplayers:
                for p2 in lplayers:
                    if p1 == p2:
                        continue
                    s1 = rng.choice(comp_arr[p1][fid], 1)[0]
                    s2 = rng.choice(comp_arr[p2][fid], 1)[0]
                    if s1 == s2:
                        won = 0.5
                    else:
                        won = abs(float(maximization) - float(s1 < s2))

                    records.append([r, p1, p2, won])
    dt_comp = pd.DataFrame.from_records(
        records, columns=["round", "p1", "p2", "outcome"]
    )
    if random_state is None:
        dt_comp = dt_comp.sample(frac=1).sort_values("round")
    else:
        dt_comp = dt_comp.sample(frac=1, random_state=random_state).sort_values(
            "round"
        )
    model = EloEstimator(key1_field="p1", key2_field="p2", timestamp_field="round").fit(
        dt_comp, dt_comp["outcome"]
    )
    model_dt = model.rating_model.to_frame()
    ratings = np.array(model_dt[np.isnan(model_dt["valid_to"])]["rating"])
    deviations = (
        model_dt.query(f"valid_from >= {nrounds * 0.95}").groupby("key")["rating"].std()
    )

    rating_dt_elo = pd.DataFrame(
        [
            ratings,
            deviations,
            *players[players.columns],
        ]
    ).transpose()
    rating_dt_elo.columns = ["Rating", "Deviation", *players.columns]
    if return_as_pandas:
        return rating_dt_elo
    return pl.from_pandas(rating_dt_elo)
