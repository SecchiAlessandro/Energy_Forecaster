# RES Generation Model Training Notebook

## Overview
This notebook implements a machine learning pipeline to forecast Renewable Energy Sources (RES) generation for the Italian energy market for the years 2025-2029. It uses historical RES generation data from 2015-2024 to train an XGBoost regression model that captures seasonal patterns and trends in renewable energy production.

## Purpose
- Train a predictive model for RES generation in Italy
- Generate daily RES generation forecasts for 2025-2029
- Analyze feature importance to understand key drivers of RES generation
- Evaluate model performance using multiple metrics
- Create visualizations for model interpretation and analysis

## Data Requirements
- Processed RES generation data from 2015-2024 (`data/final/Italy/res_generation2015_2024.csv`)
  - This file should contain daily RES generation data with columns for:
    - `date`: Date in YYYY-MM-DD format
    - `total_res_mw`: Total RES generation in megawatts
    - Source-specific columns: `solar_mw`, `wind_mw`, `hydro_mw`, `biomass_mw`, `geothermal_mw`

## Processing Steps
1. **Data Loading and Preparation**
   - Convert date to datetime and set as index
   - Create time-based features (year, month, day of week, etc.)
   - Create lag features (1, 7, 14, 30 days)
   - Create rolling window features (7, 14, 30 days)
   - Create seasonal features using sine/cosine transformations

2. **Feature Selection**
   - Time features: month, day of week, quarter, week of year
   - Cyclical features: month_sin, month_cos, day_of_year_sin, day_of_year_cos
   - Lag features: total_res_lag_1, total_res_lag_7, total_res_lag_14, total_res_lag_30
   - Rolling window statistics: means and standard deviations for 7, 14, and 30-day windows
   - Source-specific features: solar_mw, wind_mw, hydro_mw, biomass_mw, geothermal_mw

3. **Model Training**
   - Split data into training (before 2023) and validation (2023-2024) sets
   - Train XGBoost regression model with hyperparameters:
     - n_estimators: 1000
     - learning_rate: 0.05
     - max_depth: 5
     - subsample: 0.8
     - colsample_bytree: 0.8

4. **Model Evaluation**
   - Calculate performance metrics:
     - Root Mean Squared Error (RMSE)
     - Mean Absolute Error (MAE)
     - R-squared (R²)
   - Generate visualizations:
     - Actual vs. Predicted plot
     - Feature importance plot
     - Scatter plot of actual vs. predicted values

5. **Future Predictions**
   - Generate daily RES generation forecasts for 2025-2029
   - Use a rolling approach to incorporate predictions as inputs for future days
   - Apply seasonal patterns based on historical data

## Outputs
- **Trained Model**
  - XGBoost model saved to `models/res_generation_xgb_v1.joblib`
  - Feature list saved to `models/res_generation_features.joblib`

- **Predictions**
  - Future RES generation predictions saved to `data/final/Italy/res_generation2025_2029.csv`

- **Visualizations**
  - Actual vs. Predicted plot: `outputs/images/res_actual_vs_predicted.png`
  - Feature importance plot: `outputs/images/res_feature_importance.png`
  - Scatter plot of actual vs. predicted: `outputs/images/res_scatter_actual_vs_predicted.png`
  - Historical and future predictions plot: `outputs/images/res_future_predictions.png`

## Usage Instructions
1. Ensure the required data file (`res_generation2015_2024.csv`) is available in the `data/final/Italy/` directory
2. Run all cells in the notebook in sequence
3. Review the model performance metrics and visualizations
4. Check the output directories for the generated files

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

## Notes
- The model uses a time series split to ensure proper evaluation of forecasting performance
- Feature importance analysis can provide insights into the key drivers of RES generation
- The future prediction approach maintains temporal dependencies by using previous predictions as inputs for future days
- Seasonal patterns are preserved in the forecasts through cyclical encoding of time features 