"""
degradation.py

Data-quality degradation functions used in the robustness experiments.

Each function returns a degraded copy of the input DataFrame without modifying the original data. Degradation can be restricted to 
selected columns or observations where applicable.

A random_state parameter is used to make stochastic degradation reproducible across repeated experiments.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# 1. Missingness - Missing Completely At Random (MCAR)

def inject_missing_mcar(
    X: pd.DataFrame,
    pct: float,
    columns: Optional[Sequence[str]] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Randomly set a fraction of cells to NaN in the selected columns.

    Parameters -

    X : pd.DataFrame
        Input DataFrame.
    pct : float
        Fraction of cells to set to NaN.
    columns : Sequence[str], optional
        Columns to modify. Uses all columns if None.
    random_state : int, optional
        Random seed for reproducibility.

    Returns -
        pd.DataFrame
        Copy of X with missing values introduced.
    """
    if not 0 <= pct <= 1:
        raise ValueError(f"pct must be in [0, 1], got {pct}")

    rng = np.random.default_rng(random_state)
    X_deg = X.copy()
    cols = list(columns) if columns is not None else list(X.columns)

    for col in cols:
        mask = rng.random(len(X_deg)) < pct
        X_deg.loc[mask, col] = np.nan

    return X_deg

# 2. Missingness - Missing Not At Random (MNAR)

def inject_missing_mnar(
    X: pd.DataFrame,
    target_col: str,
    condition: pd.Series,
    p_high: float = 0.4,
    p_low: float = 0.05,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Introduce condition-dependent missingness in a single column.

    Rows where condition is True use p_high as the missingness probability; all other rows use p_low.

    Parameters -

    X : pd.DataFrame
        Input DataFrame.
    target_col : str
        Column in which to introduce missing values.
    condition : pd.Series
        Boolean Series aligned with X.
    p_high : float
        Missingness probability where condition is True.
    p_low : float
        Missingness probability where condition is False.
    random_state : int, optional
        Random seed for reproducibility.

    Returns -
    
    pd.DataFrame
        Copy of X with missing values introduced in target_col.
    """
    if target_col not in X.columns:
        raise ValueError(f"{target_col!r} not found in X.columns")
    if not condition.index.equals(X.index):
        raise ValueError("condition must share X's index")

    rng = np.random.default_rng(random_state)
    X_deg = X.copy()

    probs = np.where(condition.values, p_high, p_low)
    draws = rng.random(len(X_deg))
    missing_mask = draws < probs

    X_deg.loc[missing_mask, target_col] = np.nan
    return X_deg

# 3. Feature noise (numeric columns)

def inject_noise(
    X: pd.DataFrame,
    columns: Sequence[str],
    std_frac: float = 0.1,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add Gaussian noise to selected numeric columns.

    Noise is scaled using each column's standard deviation.

    Parameters -

    X : pd.DataFrame
        Input DataFrame.
    columns : Sequence[str]
        Numeric columns to which noise is added. Pass only genuinely continuous/count features (e.g. HYEARGRx), adding noise to
        category-coded columns does not correspond to any realistic data-quality issue.
    std_frac : float
        Noise standard deviation as a fraction of the column standard deviation.
    random_state : int, optional 
        Random seed for reproducibility.

    Returns -

    pd.DataFrame
        Copy of X with noise added to the selected columns.
    """
    rng = np.random.default_rng(random_state)
    X_deg = X.copy()

    for col in columns:
        if not pd.api.types.is_numeric_dtype(X_deg[col]):
            raise ValueError(f"Column {col!r} is not numeric; inject_noise "
                              f"is only meant for continuous/count features")
        col_std = X_deg[col].std()
        noise = rng.normal(loc=0.0, scale=std_frac * col_std, size=len(X_deg))
        X_deg[col] = X_deg[col] + noise

    return X_deg

# 4. Categorical corruption (data entry error simulation)

def inject_corruption(
    X: pd.DataFrame,
    columns: Sequence[str],
    pct: float,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Randomly corrupt values in selected categorical columns.

    A proportion of values in each column is replaced with a different value drawn from that column's observed values.

    Note: clean_data.csv has no `category`-dtype columns, so corrupted values here still pass through the same numeric 
    (imputer + StandardScaler) preprocessing branch as the baseline's real inputs, no separate handling is needed.

    Parameters -

    X : pd.DataFrame
        Input DataFrame.
    columns : Sequence[str]
        Categorical columns to corrupt.
    pct : float
        Fraction of values in each column to corrupt.
    random_state : int, optional
        Random seed for reproducibility.

    Returns -

    pd.DataFrame
        Copy of X with corrupted values in the selected columns.
    """
    if not 0 <= pct <= 1:
        raise ValueError(f"pct must be in [0, 1], got {pct}")

    rng = np.random.default_rng(random_state)
    X_deg = X.copy()

    for col in columns:
        unique_vals = X_deg[col].dropna().unique()
        if len(unique_vals) < 2:
            continue  # nothing to reassign to

        mask = rng.random(len(X_deg)) < pct
        idx_to_corrupt = X_deg.index[mask]

        for idx in idx_to_corrupt:
            current_val = X_deg.at[idx, col]
            choices = unique_vals[unique_vals != current_val]
            if len(choices) == 0:
                continue
            X_deg.at[idx, col] = rng.choice(choices)

    return X_deg

# 5. Degradation summary

def summarize_degradation(X_before: pd.DataFrame, X_after: pd.DataFrame) -> dict:
    """
    Compare a clean DataFrame to its degraded counterpart and report, per column, the fraction of cells that became 
    NaN and the fraction of non-NaN cells whose value changed. Useful as a logged sanity-check alongside every experiment run, 
    so results tables can cite the actual realized degradation rate, not just the requested one.
    """
    rows = []
    for col in X_before.columns:
        before = X_before[col]
        after = X_after[col]

        newly_missing = (after.isna() & ~before.isna()).mean()
        both_present = ~before.isna() & ~after.isna()
        if both_present.sum() > 0:
            changed = (before[both_present] != after[both_present]).mean()
        else:
            changed = np.nan

        rows.append({
            "column": col,
            "pct_newly_missing": round(newly_missing, 4),
            "pct_value_changed": round(changed, 4) if not np.isnan(changed) else 0.0,
        })

    return pd.DataFrame(rows).set_index("column")


if __name__ == "__main__":
    # Lightweight self-test using the real COM-2 feature columns, so this
    # module can be sanity-checked without needing xgboost/joblib installed.
    print("Running degradation.py self-test on clean_data.csv ...")

    df = pd.read_csv("clean_data.csv")
    census_cols = [
        'hhsizex', 'sft', 'nssech9', 'nssecp9', 'hhtype11', 'ager',
        'sexhrp', 'hhcomp1', 'ndepchild', 'hhltsick', 'tenure2', 'prevten',
        'tenex', 'tenure4x', 'bedrqx', 'DWtype', 'fuelx',
    ]
    other_cols = ['HYEARGRx', 'FloorArea', 'housex']
    X = df[census_cols + other_cols].copy()

    print("\n--- MCAR missingness, 20% ---")
    X_mcar = inject_missing_mcar(X, pct=0.20, random_state=1)
    print(summarize_degradation(X, X_mcar).head())

    print("\n--- MNAR missingness on HYEARGRx, renters vs owners ---")
    is_renter = X['tenure4x'] == 2
    X_mnar = inject_missing_mnar(
        X, target_col='HYEARGRx', condition=is_renter,
        p_high=0.4, p_low=0.05, random_state=1,
    )
    print(summarize_degradation(X, X_mnar).loc[['HYEARGRx']])
    print(f"Realized rate | renters: {X_mnar.loc[is_renter, 'HYEARGRx'].isna().mean():.3f}"
          f" | owners: {X_mnar.loc[~is_renter, 'HYEARGRx'].isna().mean():.3f}")

    print("\n--- Gaussian noise on HYEARGRx and FloorArea, std_frac=0.1 ---")
    X_noise = inject_noise(X, columns=['HYEARGRx', 'FloorArea'], std_frac=0.1, random_state=1)
    print(summarize_degradation(X, X_noise).loc[['HYEARGRx', 'FloorArea']])

    print("\n--- Categorical corruption on tenure2, fuelx, 15% ---")
    X_corrupt = inject_corruption(X, columns=['tenure2', 'fuelx'], pct=0.15, random_state=1)
    print(summarize_degradation(X, X_corrupt).loc[['tenure2', 'fuelx']])

    print("\nSelf-test complete.")
