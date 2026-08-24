"""
reproduce_and_save_rf_baseline.py

Purpose:
Fit and save a Random Forest baseline using the same COM-2 features, train/test split and preprocessing used for the XGBoost model.

The Random Forest uses class_weight="balanced", consistent with the baseline comparison in Zheng and McKenna. It is kept untuned to 
match the original comparison.

It's fitted independently here rather than extracted from their notebook, since their comparison RF isn't saved there as a standalone, 
loadable artifact.

Usage:
python reproduce_and_save_rf_baseline.py

Outputs (written to ./artifacts/):
    - rf_pipeline.joblib  : fitted sklearn Pipeline
    - rf_metadata.joblib  : model metrics, threshold and features
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score
)

import joblib

RANDOM_STATE = 42
DATA_PATH = "clean_data.csv"
ARTIFACT_DIR = Path("./artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

list_census_proxy = [
    'hhsizex', 'sft', 'nssech9', 'nssecp9', 'hhtype11', 'ager',
    'sexhrp', 'hhcomp1', 'ndepchild', 'hhltsick', 'tenure2', 'prevten',
    'tenex', 'tenure4x', 'bedrqx', 'DWtype', 'fuelx',
]
list_other_inputs_proxy = ['HYEARGRx', 'FloorArea', 'housex']
TARGET = 'fpLIHCflg'
COM2_FEATURES = list_census_proxy + list_other_inputs_proxy


def load_data_and_split(data_path: str = DATA_PATH):
    df = pd.read_csv(data_path)
    df = df[list_census_proxy + list_other_inputs_proxy + ['sap12', TARGET]]
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()
    Train_x, Test_x, Train_y, Test_y = train_test_split(
        X, y, stratify=y, random_state=RANDOM_STATE
    )
    return Train_x, Test_x, Train_y, Test_y


def build_preprocessor():
    # Match the baseline preprocessing so differences between models
    # are not caused by different preprocessing steps.
    num_pipe = make_pipeline(
        SimpleImputer(strategy="mean", add_indicator=True),
        StandardScaler(),
    )
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return make_column_transformer(
        (num_pipe, selector(dtype_include="number")),
        (cat_pipe, selector(dtype_include="category")),
        n_jobs=-1,
    )


def main():
    print("Loading data and reproducing the train/test split (random_state=42)...")
    Train_x, Test_x, Train_y, Test_y = load_data_and_split()

    X_train = Train_x[COM2_FEATURES].copy()
    y_train = Train_y.copy()
    X_test = Test_x[COM2_FEATURES].copy()
    y_test = Test_y.copy()
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")

    preprocessor = build_preprocessor()
    rf_pipeline = make_pipeline(
        preprocessor,
        RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    )

    print("Fitting RF pipeline...")
    rf_pipeline.fit(X_train, y_train)

    probs = rf_pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)  # default threshold; adjust if you
                                          # want to match COM-2's 0.45 convention
    metrics = {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Balanced Accuracy": round(balanced_accuracy_score(y_test, preds), 4),
        "ROC-AUC": round(roc_auc_score(y_test, probs), 4),
    }
    print("RF metrics at threshold 0.5:", metrics)

    print("Saving artifacts to", ARTIFACT_DIR.resolve())
    joblib.dump(rf_pipeline, ARTIFACT_DIR / "rf_pipeline.joblib")
    joblib.dump(
        {
            "threshold": 0.5,
            "features": COM2_FEATURES,
            "metrics": metrics,
            "random_state": RANDOM_STATE,
            "note": "Untuned RF with class_weight='balanced', for comparison "
                    "against the tuned XGBoost COM-2 pipeline.",
        },
        ARTIFACT_DIR / "rf_metadata.joblib",
    )
    print("Done.")


if __name__ == "__main__":
    main()
