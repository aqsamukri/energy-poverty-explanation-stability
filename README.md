# Trustworthy AI for Energy Poverty Risk Assessment

Code, data, and results for the dissertation *"Trustworthy AI for Energy Poverty Risk Assessment: Evaluating the Stability of Explainable Machine Learning Under Real-World Data Quality Challenges."* Extends the pipeline from Zheng & McKenna (2025) with a controlled data-degradation framework, testing whether SHAP explanations stay stable as input data quality deteriorates.

## Attribution

Builds directly on the pipeline published in:

> Zheng, L. & McKenna, E. (2025). Machine Learning with Administrative Data
> for Energy Poverty Identification in the UK. *Energies*, 18(12), 3054.
> Original code (MIT licence): https://github.com/linzzuk/Energy_Poverty_Prediction_paper_EHS_data

Everything beyond the reproduced baseline - the degradation framework, the stability metrics, the experiments, is original extension work, not the original authors' code.

## Repository Structure

```
├── artifacts/                          # Fitted models (XGBoost/COM-2 + Random Forest)
├── results/                            # Experiment outputs: CSVs + figures
├── clean_data.csv                      # Processed dataset (from Zheng & McKenna)
├── degradation.py                      # Four data degradation mechanisms
├── experiment_utils.py                 # Seeding, bootstrap CIs, results logging
├── shap_stability.py                   # SHAP computation + stability metrics
├── reproduce_and_save_rf_baseline.py   # Random Forest comparison model
├── household_case_study.py             # Household-level case study
├── run_robustness_experiments.ipynb    # Main experiment notebook
└── requirements.txt
```

## Dependencies

pip install -r requirements.txt


## How to Use

1. **Run the main experiment notebook**: `run_robustness_experiments.ipynb` loads the already-fitted models from `artifacts/` and produces the stability tables and figures.
2. **Run the case study**: `household_case_study.py` generates the household-level SHAP comparison.
3. No retraining or external data access required, `artifacts/` contains the exact fitted models behind every reported result.

## Key Features

- **Four degradation mechanisms**: MCAR and MNAR missingness, Gaussian feature noise, categorical corruption
- **Two-level stability evaluation**: global feature-ranking stability and instance-level explanation stability
- **Model comparison**: XGBoost (COM-2) vs. Random Forest under identical degradation conditions
- **Statistical rigor**: 50 seeded repeats per condition, reported with 95% bootstrap confidence intervals

## Model Provenance

The COM-2 (XGBoost) model was produced by adding a save step to the end of Zheng & McKenna's own `ep_prediction_model.ipynb` and running it — their original notebook isn't redistributed here, only referenced above. The Random Forest comparison model is fitted independently in `reproduce_and_save_rf_baseline.py`, included here in full.
