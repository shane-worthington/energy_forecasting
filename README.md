# Electricity Demand Forecasting: Model Benchmarking Code

Source code for a dissertation benchmarking statistical, machine learning and deep learning models for one-hour-ahead electricity demand forecasting at two network scales: a national demand series (Spain) and a Western Sydney zone substation.

Five models are compared under a single one-step-ahead convention and a scale-free error measure (MASE): ARIMA, SARIMA, XGBoost, an LSTM network, and a fixed 50/50 SARIMA + XGBoost hybrid.

## Data and de-identification

The substation demand data was provided by Endeavour Energy and has been **de-identified** for release: the specific site name and its coordinates are removed from the code, and the data files are referred to by generic names. **The demand and weather data are not included in this repository**, so the scripts are provided for transparency of method rather than as a turnkey reproduction package. The national (Spain) data is public (Kaggle / ENTSO-E) and is cited in the dissertation.

## Environment

Python 3.10, with `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `pmdarima`, `xgboost`, `tensorflow`, `scipy` and `matplotlib`. The exact package versions used to produce the reported results are listed in the dissertation (Section 3.11). A fixed random seed of 42 is set wherever a step involves randomness.

## How it fits together

`data_loader.py` is the shared entry point: it loads, cleans and splits both datasets into one common schema, so every model sees identical data. Each model script writes a forecast file; `unified_comparison.py` then aligns those forecasts on a common test window and computes the final comparable metrics.

## Files

| File | Purpose | Dissertation |
|------|---------|--------------|
| `data_loader.py` | Load, clean and split both datasets | Sections 3.2, 3.6 |
| `dataset_summary.py` | Descriptive statistics | Tables 3.1, 3.2 |
| `cross_dataset_smoothness.py` | Load smoothness metrics | Section 4.5 |
| `arima_model.py` | ARIMA and SARIMA | Sections 3.8.1 to 3.8.2 |
| `xgboost_model.py` | XGBoost | Section 3.8.3 |
| `lstm_model.py` | LSTM | Section 3.8.4 |
| `hybrid_model.py` | SARIMA + XGBoost hybrid | Section 3.8.5 |
| `evaluation.py` | Error metrics (MAE, RMSE, MAPE, MASE, R²) | Section 3.9 |
| `unified_comparison.py` | Common-window alignment and metrics | Section 3.10 |
| `dm_test.py` | Diebold-Mariano test | Section 4.7 |
| `hybrid_analysis.py` | Hybrid weight sweep | Section 4.6 |
