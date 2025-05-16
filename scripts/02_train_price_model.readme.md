# Price Forecasting Model Training Script

## Overview
This script trains an XGBoost regression model to forecast energy prices for the Italian market for the years 2025-2029. It uses historical price data from 2015-2024 to train a model that captures seasonal patterns and trends in energy prices.

## Purpose
- Train a predictive model for energy price forecasting in Italy
- Generate daily price forecasts for 2025-2029
- Analyze feature importance to understand key drivers of price fluctuations
- Evaluate model performance using multiple metrics
- Create visualizations for model interpretation and analysis

## Data Requirements
- Processed price data from 2015-2024 (`data/final/Italy/energy_price2015_2024.csv`)
- Train/test split datasets (optional, will be created if not found):
  - `data/final/Italy/price_train_data.csv`
  - `data/final/Italy/price_test_data.csv`

## Processing Steps
1. **Data Loading**
   - Load processed price data with engineered features
   - Use existing train/test split if available, or create new split (80% train, 20% test)

2. **Feature Engineering** (if needed)
   - Create time-based features (day of week, month, quarter, year, etc.)
   - Generate lag features (1-day, 7-day, 30-day)
   - Calculate rolling window statistics (7-day and 30-day moving averages)
   - Handle missing values from lag features

3. **Model Training**
   - Prepare features and target variables
   - Train XGBoost regression model with optimized hyperparameters
   - Evaluate model on test set
   - Save trained model and feature list

4. **Model Evaluation**
   - Calculate performance metrics (RMSE, MAE, R²)
   - Generate visualizations:
     - Actual vs. predicted plot
     - Scatter plot of actual vs. predicted values
     - Feature importance plot

5. **Future Predictions**
   - Create date range for 2025-2029
   - Generate time-based features for future dates
   - Use rolling prediction approach to forecast prices day by day
   - Save predictions to CSV file
   - Create visualization of historical and future prices

## Outputs
- **Model Files**:
  - Trained XGBoost model: `models/energy_price_xgb_v1.joblib`
  - Feature list: `models/price_features.joblib`

- **Data Files**:
  - Future price predictions: `data/final/Italy/energy_price2025_2029.csv`
  - Train/test datasets (if created): 
    - `data/final/Italy/price_train_data.csv`
    - `data/final/Italy/price_test_data.csv`

- **Visualizations**:
  - `outputs/images/price_actual_vs_predicted.png`: Time series of actual vs. predicted prices
  - `outputs/images/price_scatter_actual_vs_predicted.png`: Scatter plot of actual vs. predicted prices
  - `outputs/images/price_feature_importance.png`: Feature importance plot
  - `outputs/images/price_future_predictions.png`: Historical and future price predictions

## Usage Instructions
1. Ensure the processed price data file (`energy_price2015_2024.csv`) is available in the `data/final/Italy/` directory
2. Run the script: `python scripts/02_train_price_model.py`
3. Review the terminal output for model performance metrics
4. Examine the generated visualizations in the `outputs/images/` directory
5. Use the predictions in `data/final/Italy/energy_price2025_2029.csv` for further analysis

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib
- pathlib

## Model Details
- **Algorithm**: XGBoost Regression
- **Hyperparameters**:
  - n_estimators: 500
  - learning_rate: 0.05
  - max_depth: 5
  - subsample: 0.8
  - colsample_bytree: 0.8
  - objective: reg:squarederror
  - eval_metric: rmse
  - random_state: 42

## Key Findings
- The model achieves good performance with an R² score of approximately 0.74
- RMSE of around 12.6 €/MWh indicates reasonable prediction accuracy
- Lag features (especially price_lag1 and price_rolling_7d_mean) are typically the most important predictors
- The model successfully captures seasonal patterns in energy prices
- Future predictions follow expected seasonal patterns while accounting for historical trends

## Next Steps
- Convert the script to a Jupyter notebook for interactive analysis
- Explore additional feature engineering approaches to improve model performance
- Consider ensemble methods combining multiple models
- Analyze the predictions in conjunction with demand and RES generation forecasts 