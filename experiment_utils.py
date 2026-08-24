"""
experiment_utils.py

Purpose:
Utility functions used across the robustness experiments.

The module provides functions for repeated experiment runs, reproducible random seeds, confidence intervals, 
and saving experiment results.

It is independent of the degradation and SHAP implementations and can be used with different numerical evaluation metrics.
"""

from pathlib import Path
from typing import Callable, Sequence, Optional
import time

import numpy as np
import pandas as pd

# 1. Reproducible seed generation

def make_seeds(n_repeats: int, base_seed: int = 0) -> list:
    """
    Generate reproducible random seeds for repeated experiment runs.

    Seeds are derived from a single base_seed, rather than using range(n_repeats) directly, to avoid accidentally correlating 
    repeat index with seed value in a way that could interact with the degradation functions' own seeding.

    Parameters -

    n_repeats : int
        Number of seeds to generate.
    base_seed : int
        Base seed used to generate the sequence.

    Returns -

    list
        Generated integer seeds.
    """
    rng = np.random.default_rng(base_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n_repeats)]

# 2. Bootstrap confidence intervals

def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: Optional[int] = None,
) -> dict:
    """
    Compute a bootstrap confidence interval for the mean.

    Parameters -

    values : sequence of float
       Metric values from repeated runs.
    n_boot : int
       Number of bootstrap resamples.
    ci : float
       Confidence level.
    random_state : int, optional
       Random seed for reproducibility.

    Returns -

    dict
       Mean, standard deviation, confidence interval bounds, and number of valid values.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    if n == 1:
        return {"mean": values[0], "std": np.nan, "ci_lower": values[0],
                "ci_upper": values[0], "n": 1}

    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "ci_lower": lower,
        "ci_upper": upper,
        "n": n,
    }


def bootstrap_diff_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: Optional[int] = None,
) -> dict:
    """
    Compute a bootstrap confidence interval for the difference in means between two sets of repeated measurements.

    Parameters -

    values_a : Sequence[float]
       Values from the first condition or model.
    values_b : Sequence[float]
       Values from the second condition or model.
    n_boot : int
       Number of bootstrap resamples.
    ci : float
       Confidence level.
    random_state : int, optional
       Random seed for reproducibility.

    Returns -

    dict
       Mean difference, confidence interval bounds, and whether the interval excludes zero.
    """
    a = np.asarray(values_a, dtype=float)
    a = a[~np.isnan(a)]
    b = np.asarray(values_b, dtype=float)
    b = b[~np.isnan(b)]

    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        diffs[i] = sample_a.mean() - sample_b.mean()

    alpha = 1 - ci
    lower = np.percentile(diffs, 100 * (alpha / 2))
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    return {
        "mean_diff": a.mean() - b.mean(),
        "ci_lower": lower,
        "ci_upper": upper,
        "excludes_zero": bool(lower > 0 or upper < 0),
    }

# 3. Repeated-run harness

def run_repeated(
    run_once_fn: Callable[[int], dict],
    n_repeats: int,
    base_seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
   Run an experiment repeatedly using reproducible random seeds and collect the results in a DataFrame.

   Parameters -

   run_once_fn : Callable[[int], dict]
       Function that accepts a random seed and returns a dictionary of metric values.
   n_repeats : int
       Number of repeated runs.
   base_seed : int
       Base seed used to generate the repeat seeds.
   verbose : bool
       Whether to print progress information.

   Returns -

   pd.DataFrame
      One row per repeat, including the seed and returned metrics.
    """
    seeds = make_seeds(n_repeats, base_seed=base_seed)
    rows = []
    start = time.time()

    for i, seed in enumerate(seeds):
        result = run_once_fn(seed)
        result = {"repeat": i, "seed": seed, **result}
        rows.append(result)
        if verbose and (i + 1) % max(1, n_repeats // 5) == 0:
            print(f"  ... {i + 1}/{n_repeats} repeats done "
                  f"({time.time() - start:.1f}s elapsed)")

    return pd.DataFrame(rows)

# 4. Structured results logging

def save_results(
    df: pd.DataFrame,
    results_dir: str,
    experiment_name: str,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save experiment results to a timestamped CSV file.

    If metadata is provided, it is saved as a separate CSV using the same timestamp.

    Parameters -

    df : pd.DataFrame
        Results to save.
    results_dir : str
        Output directory.
    experiment_name : str
        Name used in the output filename.
    metadata : dict, optional
        Additional experiment metadata to save.

    Returns -

    Path
        Path to the saved results CSV.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    data_path = out_dir / f"{experiment_name}_{timestamp}.csv"
    df.to_csv(data_path, index=False)

    if metadata is not None:
        meta_path = out_dir / f"{experiment_name}_{timestamp}_meta.csv"
        pd.DataFrame([metadata]).to_csv(meta_path, index=False)

    return data_path


if __name__ == "__main__":
    # Self-test with synthetic data, again no dependency on the baseline model,
    # so that this can be checked before the full experiment pipeline exists.
    print("Running experiment_utils.py self-test ...")

    seeds = make_seeds(5, base_seed=42)
    print("Seeds:", seeds)
    print("Same base_seed reproduces same seeds:",
          seeds == make_seeds(5, base_seed=42))

    fake_stability_scores = [0.91, 0.89, 0.93, 0.90, 0.88, 0.92, 0.87]
    ci_result = bootstrap_ci(fake_stability_scores, random_state=1)
    print("\nBootstrap CI on fake stability scores:", ci_result)

    fake_group_a = [0.91, 0.89, 0.93, 0.90]   # e.g. XGBoost stability
    fake_group_b = [0.75, 0.70, 0.78, 0.72]   # e.g. RF stability
    diff_result = bootstrap_diff_ci(fake_group_a, fake_group_b, random_state=1)
    print("\nBootstrap diff CI (A - B):", diff_result)

    def fake_run_once(seed):
        rng = np.random.default_rng(seed)
        return {"stability_score": rng.uniform(0.8, 0.95)}

    df_repeats = run_repeated(fake_run_once, n_repeats=10, base_seed=1)
    print("\nrun_repeated output:\n", df_repeats)

    saved_path = save_results(
        df_repeats, results_dir="./_selftest_results",
        experiment_name="selftest",
        metadata={"n_repeats": 10, "base_seed": 1, "note": "self-test only"},
    )
    print("\nSaved to:", saved_path)
    print("\nSelf-test complete.")
