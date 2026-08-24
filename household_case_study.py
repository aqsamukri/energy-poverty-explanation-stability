"""
household_case_study.py

Purpose:
Examine SHAP explanation changes for an individual household under data degradation.

The default condition is 30% MCAR degradation, chosen deliberately, since Section 3.5 found this is where predictive performance 
degraded most severely. The script reports the number of prediction changes and gives a detailed comparison for one household, 
including changes in feature values, SHAP contributions and predicted classification.

A comparison of the household's main SHAP contributions before and after degradation is also saved to the results folder.

Usage:
    python household_case_study.py

In Jupyter:
    %run household_case_study.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from degradation import inject_missing_mcar, inject_corruption, inject_noise, inject_missing_mnar
from shap_stability import compute_shap_values, instance_cosine_similarity

# Configuration

ARTIFACT_DIR = "./artifacts"
RESULTS_DIR = "./results"
DEGRADATION_TYPE = "MCAR"   # "MCAR", "Corruption", "Noise", or "MNAR"
SEVERITY = 0.30
SEED = 2026
SHAP_SAMPLE_SIZE = 100
CORRUPTION_COLUMNS = ["tenure2", "fuelx", "DWtype"]
NOISE_COLUMNS = ["HYEARGRx", "FloorArea"]

Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)


def degrade(X, degradation_type, severity, seed):
    if degradation_type == "MCAR":
        return inject_missing_mcar(X, pct=severity, random_state=seed)
    elif degradation_type == "Corruption":
        return inject_corruption(X, columns=CORRUPTION_COLUMNS, pct=severity, random_state=seed)
    elif degradation_type == "Noise":
        return inject_noise(X, columns=NOISE_COLUMNS, std_frac=severity, random_state=seed)
    elif degradation_type == "MNAR":
        is_renter = X["tenure4x"] == 2
        return inject_missing_mnar(
            X, target_col="HYEARGRx", condition=is_renter,
            p_high=min(0.40 * (severity / 0.20), 0.95), p_low=0.05, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown degradation_type: {degradation_type}")


def main():
    print(f"Loading artifacts and computing SHAP for {DEGRADATION_TYPE} at severity {SEVERITY}...")

    pipeline = joblib.load(f"{ARTIFACT_DIR}/com2_pipeline.joblib")
    metadata = joblib.load(f"{ARTIFACT_DIR}/com2_metadata.joblib")
    data_splits = joblib.load(f"{ARTIFACT_DIR}/com2_data_splits.joblib")
    X_test = data_splits["X_test"]
    threshold = metadata["threshold"]

    # Clean and degraded SHAP, same 100 households in both 
    clean_result = compute_shap_values(pipeline, X_test, sample_size=SHAP_SAMPLE_SIZE)
    X_degraded = degrade(X_test, DEGRADATION_TYPE, SEVERITY, SEED)
    degraded_result = compute_shap_values(pipeline, X_degraded, sample_size=SHAP_SAMPLE_SIZE)

    sample_index = clean_result["sample_index"]  # same 100 household indices for both

    # Headline statistic: how many predictions flipped? 
    clean_raw = X_test.loc[sample_index]
    degraded_raw = X_degraded.loc[sample_index]

    clean_probs = pipeline.predict_proba(clean_raw)[:, 1]
    degraded_probs = pipeline.predict_proba(degraded_raw)[:, 1]
    clean_preds = (clean_probs >= threshold).astype(int)
    degraded_preds = (degraded_probs >= threshold).astype(int)

    flipped_mask = clean_preds != degraded_preds
    n_flipped = int(flipped_mask.sum())
    print(f"\n=== HEADLINE STATISTIC ===")
    print(f"{n_flipped} out of {SHAP_SAMPLE_SIZE} households' predicted classification "
          f"flipped under {DEGRADATION_TYPE} at severity {SEVERITY}.")
    if n_flipped > 0:
        flipped_to_missed = int(((clean_preds == 1) & (degraded_preds == 0)).sum())
        flipped_to_flagged = int(((clean_preds == 0) & (degraded_preds == 1)).sum())
        print(f"  - {flipped_to_missed} households went from AT-RISK to NOT-AT-RISK "
              f"(i.e., a genuinely vulnerable household would now be missed)")
        print(f"  - {flipped_to_flagged} households went from NOT-AT-RISK to AT-RISK")

    # Select the household for the detailed case study 
    cosine_sims = instance_cosine_similarity(clean_result["shap_values"], degraded_result["shap_values"])

    if n_flipped > 0:
        # Select the flipped household with the largest probability change.
        prob_shift = np.abs(degraded_probs - clean_probs)
        candidate_positions = np.where(flipped_mask)[0]
        case_position = candidate_positions[np.argmax(prob_shift[candidate_positions])]
        print(f"\nSelected household for case study: one whose classification FLIPPED "
              f"(largest probability shift among flipped households).")
    else:
        # Fallback: household with the lowest cosine similarity (most unstable
        # explanation), even if the classification itself didn't flip.
        case_position = int(np.argmin(cosine_sims))
        print(f"\nNo households flipped classification under this condition/severity. "
              f"Selected household for case study: lowest cosine similarity instead.")

    household_id = sample_index[case_position]
    print(f"Household index in original X_test: {household_id}")
    print(f"Cosine similarity for this household: {cosine_sims[case_position]:.3f} "
          f"(sample mean: {cosine_sims.mean():.3f})")
    print(f"Predicted probability -- clean: {clean_probs[case_position]:.3f}, "
          f"degraded: {degraded_probs[case_position]:.3f} (threshold: {threshold})")
    print(f"Classification -- clean: {'AT-RISK' if clean_preds[case_position] else 'not at-risk'}, "
          f"degraded: {'AT-RISK' if degraded_preds[case_position] else 'not at-risk'}")

    # Which raw features actually changed for this household?
    clean_row_raw = X_test.loc[[household_id]]
    degraded_row_raw = X_degraded.loc[[household_id]]
    changed_features = []
    for col in X_test.columns:
        c_val, d_val = clean_row_raw[col].values[0], degraded_row_raw[col].values[0]
        is_different = (pd.isna(c_val) != pd.isna(d_val)) or (
            not pd.isna(c_val) and not pd.isna(d_val) and c_val != d_val
        )
        if is_different:
            changed_features.append((col, c_val, d_val))
    print(f"\nRaw feature(s) actually altered for this household:")
    for col, c_val, d_val in changed_features:
        print(f"  {col}: {c_val} -> {d_val}")

    # SHAP comparison table, top 8 features by change magnitude 
    feature_names = [fn.replace("pipeline-1__", "") for fn in clean_result["feature_names"]]
    clean_shap_row = clean_result["shap_values"][case_position]
    degraded_shap_row = degraded_result["shap_values"][case_position]

    comparison_df = pd.DataFrame({
        "feature": feature_names,
        "clean_shap": clean_shap_row,
        "degraded_shap": degraded_shap_row,
    })
    comparison_df["change"] = comparison_df["degraded_shap"] - comparison_df["clean_shap"]
    comparison_df["abs_change"] = comparison_df["change"].abs()
    comparison_df = comparison_df.sort_values("abs_change", ascending=False)

    print(f"\nTop 8 features by SHAP value change for household {household_id}:")
    print(comparison_df[["feature", "clean_shap", "degraded_shap", "change"]].head(8).to_string(index=False))

    # Figure: before/after SHAP bar chart 
    top8 = comparison_df.head(8).sort_values("clean_shap")
    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(top8))
    width = 0.35
    ax.barh(y_pos - width / 2, top8["clean_shap"], height=width, label="Clean", color="#4C72B0")
    ax.barh(y_pos + width / 2, top8["degraded_shap"], height=width, label="Degraded", color="#DD8452")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top8["feature"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on predicted energy-poverty risk)")
    ax.set_title(f"Household {household_id}: explanation before vs. after\n"
                 f"{DEGRADATION_TYPE} degradation (severity {SEVERITY})")
    ax.legend()
    fig.tight_layout()
    out_path = f"{RESULTS_DIR}/household_case_study_{DEGRADATION_TYPE}_{SEVERITY}.png"
    fig.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
