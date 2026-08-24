"""
shap_stability.py

SHAP stability metrics for the data degradation experiments.

This module compares SHAP explanations produced from clean and degraded test data. Stability is evaluated at two levels:

1. Global stability: changes in overall feature importance rankings.
2. Instance-level stability: changes in explanations for individual observations.

SHAP values are calculated using the fitted preprocessing and model steps from the baseline pipeline, saved from the modified copy of 
Zheng & McKenna's ep_prediction_model.ipynb (see README). The pipeline is not refitted on degraded data, since refitting would let 
imputation differences, not the degradation itself, explain any shift in explanations.

For instance-level comparisons, clean and degraded SHAP values must be calculated for the same observations in the same order.
"""

from typing import Optional, Sequence
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

# 1. SHAP computation (replicates the baseline's exact transform path)

def compute_shap_values(
    pipeline,
    X: pd.DataFrame,
    sample_size: int = 100,
    sample_method: str = "first",
    random_state: Optional[int] = None,
):
    """
    Compute SHAP values for a sample of observations using the fitted preprocessor and model from the baseline pipeline.

    Parameters - 

    pipeline
        Fitted sklearn pipeline containing preprocessing and model steps.
    X : pd.DataFrame
        Input data with the same feature columns used to fit the pipeline.
    sample_size : int
        Number of observations to explain.
    sample_method : str
        Sampling method. Use "first" for paired clean/degraded comparisons or "random" for exploratory analysis.
    random_state : int, optional
        Random seed used when sample_method="random".

    Returns - 

    dict
       SHAP values, transformed feature names, transformed observations, and the indices of the sampled observations.
    """
    import shap  # imported here so this module can be imported without
                 # requiring shap to be installed

    if sample_method == "first":
        X_sample_raw = X.iloc[:sample_size]
    elif sample_method == "random":
        X_sample_raw = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    else:
        raise ValueError(f"sample_method must be 'first' or 'random', got {sample_method!r}")

    sample_index = X_sample_raw.index

    # Access the fitted preprocessing and model steps by position so the
    # function works with both pipelines used in this project.
    preprocessor = pipeline.steps[0][1]
    model = pipeline.steps[-1][1]

    X_transformed_array = preprocessor.transform(X_sample_raw)
    feature_names = list(preprocessor.get_feature_names_out())
    X_sample_transformed = pd.DataFrame(X_transformed_array, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X_sample_transformed)

    shap_values = shap_explanation.values
    if shap_values.ndim == 3:
        # Use class 1 (energy-poor) when SHAP returns
        # separate values for each class.
        shap_values = shap_values[:, :, 1]

    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "X_sample_transformed": X_sample_transformed,
        "sample_index": sample_index,
    }

# 2. Global stability: did the overall feature-importance ranking change?

def mean_abs_importance(shap_values: np.ndarray) -> np.ndarray:
    """
    Calculate global feature importance using the mean absolute SHAP value for each feature.

    Parameters -
    shap_values : np.ndarray
        SHAP values with shape (n_instances, n_features).

    Returns - 
    np.ndarray
        Mean absolute SHAP value for each feature.
    """
    return np.abs(shap_values).mean(axis=0)


def global_rank_correlation(
    shap_clean: np.ndarray,
    shap_degraded: np.ndarray,
    method: str = "spearman",
) -> dict:
    """
    Compare global SHAP feature rankings between clean and degraded data.

    Feature importance is calculated using the mean absolute SHAP value for each feature. Spearman or Kendall rank correlation
    is then used to compare the rankings.

    Parameters -

    shap_clean : np.ndarray 
        SHAP values for the clean data.
    shap_degraded : np.ndarray
        SHAP values for the degraded data.
    method : str
        Rank correlation method ("spearman" or "kendall").

    Returns -

    dict
       Correlation, p-value, and feature importance values for the clean and degraded data.   
    """
    if shap_clean.shape[1] != shap_degraded.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: clean has {shap_clean.shape[1]} "
            f"features, degraded has {shap_degraded.shape[1]}. Both must be "
            f"computed with the same pipeline/feature set."
        )

    imp_clean = mean_abs_importance(shap_clean)
    imp_degraded = mean_abs_importance(shap_degraded)

    if method == "spearman":
        corr, p_value = spearmanr(imp_clean, imp_degraded)
    elif method == "kendall":
        corr, p_value = kendalltau(imp_clean, imp_degraded)
    else:
        raise ValueError(f"method must be 'spearman' or 'kendall', got {method!r}")

    return {
        "correlation": corr,
        "p_value": p_value,
        "importance_clean": imp_clean,
        "importance_degraded": imp_degraded,
    }


def topk_jaccard_overlap(
    shap_clean: np.ndarray,
    shap_degraded: np.ndarray,
    feature_names: Sequence[str],
    k: int = 5,
) -> dict:
    """
    Compare the top-k SHAP features between clean and degraded data using Jaccard overlap.

    Parameters -

    shap_clean : np.ndarray
        SHAP values for the clean data.
    shap_degraded : np.ndarray
        SHAP values for the degraded data.
    feature_names : Sequence[str]
        Names of the features corresponding to the SHAP values.
    k : int
        Number of top features to compare.

    Returns -

    dict
    Jaccard overlap and the top-k feature sets for the clean and degraded data.
    """
    imp_clean = mean_abs_importance(shap_clean)
    imp_degraded = mean_abs_importance(shap_degraded)

    top_clean = set(np.array(feature_names)[np.argsort(imp_clean)[::-1][:k]])
    top_degraded = set(np.array(feature_names)[np.argsort(imp_degraded)[::-1][:k]])

    intersection = top_clean & top_degraded
    union = top_clean | top_degraded
    jaccard = len(intersection) / len(union) if len(union) > 0 else np.nan

    return {
        "jaccard": jaccard,
        "top_k_clean": sorted(top_clean),
        "top_k_degraded": sorted(top_degraded),
        "only_in_clean": sorted(top_clean - top_degraded),
        "only_in_degraded": sorted(top_degraded - top_clean),
    }

# 3. Instance-level stability: did this household's explanation change?

def _check_paired_shapes(shap_clean: np.ndarray, shap_degraded: np.ndarray) -> None:
    if shap_clean.shape != shap_degraded.shape:
        raise ValueError(
            f"Instance-level comparison requires identical shapes (same "
            f"households, same features). Got clean {shap_clean.shape} vs "
            f"degraded {shap_degraded.shape}. Make sure both were computed "
            f"with the same sample_size and sample_method='first' on X_test "
            f"copies that share the same row order/index."
        )


def instance_cosine_similarity(
    shap_clean: np.ndarray,
    shap_degraded: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between clean and degraded SHAP values for each observation.

    Parameters -

    shap_clean : np.ndarray
        SHAP values for the clean data.
    shap_degraded : np.ndarray
        SHAP values for the degraded data, using the same observations in the same order.

    Returns -
    
    np.ndarray
    Cosine similarity score for each observation.
    """
    _check_paired_shapes(shap_clean, shap_degraded)

    dot = np.sum(shap_clean * shap_degraded, axis=1)
    norm_clean = np.linalg.norm(shap_clean, axis=1)
    norm_degraded = np.linalg.norm(shap_degraded, axis=1)

    denom = norm_clean * norm_degraded
    # Avoid division by zero when a SHAP vector has zero magnitude.
    similarity = np.divide(
        dot, denom, out=np.zeros_like(dot, dtype=float), where=denom != 0
    )
    return similarity


def instance_rank_correlation(
    shap_clean: np.ndarray,
    shap_degraded: np.ndarray,
    method: str = "spearman",
) -> np.ndarray:
    """
    Calculate the rank correlation between clean and degraded SHAP values for each observation.

    Parameters -

    shap_clean : np.ndarray
        SHAP values for the clean data.
    shap_degraded : np.ndarray
        SHAP values for the degraded data, using the same observations in the same order.
    method : str
        Rank correlation method ("spearman" or "kendall").

    Returns -

    np.ndarray
    Rank correlation for each observation. Returns NaN where the correlation cannot be calculated.
    """
    _check_paired_shapes(shap_clean, shap_degraded)

    n_instances = shap_clean.shape[0]
    correlations = np.empty(n_instances)

    corr_fn = spearmanr if method == "spearman" else kendalltau
    if method not in ("spearman", "kendall"):
        raise ValueError(f"method must be 'spearman' or 'kendall', got {method!r}")

    for i in range(n_instances):
        row_clean = shap_clean[i, :]
        row_degraded = shap_degraded[i, :]
        if np.all(row_clean == row_clean[0]) or np.all(row_degraded == row_degraded[0]):
            correlations[i] = np.nan  # constant vector -> correlation undefined
            continue
        corr, _ = corr_fn(row_clean, row_degraded)
        correlations[i] = corr

    return correlations

# 4. Combined stability metrics

def stability_report(
    shap_clean: np.ndarray,
    shap_degraded: np.ndarray,
    feature_names: Sequence[str],
    k: int = 5,
) -> dict:
    """
    Calculate the global and instance-level SHAP stability metrics.

    Parameters -

    shap_clean : np.ndarray
        SHAP values for the clean data.
    shap_degraded : np.ndarray
        SHAP values for the degraded data.
    feature_names : Sequence[str]
        Names of the features corresponding to the SHAP values.
    k : int
        Number of top features used for Jaccard overlap.

    Returns -

    dict
    Calculated SHAP stability metrics.
    """
    global_spearman = global_rank_correlation(shap_clean, shap_degraded, method="spearman")
    global_kendall = global_rank_correlation(shap_clean, shap_degraded, method="kendall")
    jaccard = topk_jaccard_overlap(shap_clean, shap_degraded, feature_names, k=k)

    cosine_sims = instance_cosine_similarity(shap_clean, shap_degraded)
    rank_corrs = instance_rank_correlation(shap_clean, shap_degraded, method="spearman")

    return {
        "global_spearman": global_spearman["correlation"],
        "global_spearman_p": global_spearman["p_value"],
        "global_kendall": global_kendall["correlation"],
        "jaccard_top_k": jaccard["jaccard"],
        "instance_cosine_mean": float(np.mean(cosine_sims)),
        "instance_cosine_std": float(np.std(cosine_sims)),
        "instance_rank_corr_mean": float(np.nanmean(rank_corrs)),
        "instance_rank_corr_n_valid": int(np.sum(~np.isnan(rank_corrs))),
    }


if __name__ == "__main__":
    # Self-test using synthetic SHAP values.
    print("Running shap_stability.py self-test (synthetic SHAP arrays) ...")
    rng = np.random.default_rng(0)

    n_instances, n_features = 50, 6
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Create synthetic SHAP values with decreasing feature importance.
    base_scale = np.array([5, 4, 3, 2, 1, 0.5])
    shap_clean = rng.normal(loc=0, scale=base_scale, size=(n_instances, n_features))

    print("\n--- Case A: identical arrays (expect near-perfect stability) ---")
    report_identical = stability_report(shap_clean, shap_clean.copy(), feature_names, k=3)
    for key, val in report_identical.items():
        print(f"  {key}: {val}")

    print("\n--- Case B: top-2 features swap importance (expect reduced global stability) ---")
    shap_swapped = shap_clean.copy()
    shap_swapped[:, [0, 1]] = shap_swapped[:, [1, 0]]
    report_swapped = stability_report(shap_clean, shap_swapped, feature_names, k=3)
    for key, val in report_swapped.items():
        print(f"  {key}: {val}")

    print("\n--- Case C: degraded = unrelated random noise (expect low stability) ---")
    shap_random = rng.normal(loc=0, scale=1.0, size=(n_instances, n_features))
    report_random = stability_report(shap_clean, shap_random, feature_names, k=3)
    for key, val in report_random.items():
        print(f"  {key}: {val}")

    assert report_identical["global_spearman"] > 0.99, "Identical arrays should show ~perfect global correlation"
    assert report_identical["instance_cosine_mean"] > 0.99, "Identical arrays should show ~perfect cosine similarity"
    assert report_identical["jaccard_top_k"] == 1.0, "Identical arrays should show perfect top-k overlap"
    assert report_random["instance_cosine_mean"] < report_identical["instance_cosine_mean"], \
        "Random noise should be less stable than identical arrays"
    assert report_random["global_spearman"] < report_identical["global_spearman"], \
        "Random noise should show weaker global ranking correlation than identical arrays"

    print("\nAll self-test assertions passed.")
