# Energy Demand Forecasting Model Training

This notebook trains an XGBoost regression model to forecast energy demand based on historical patterns and features. The model is designed to predict daily electricity demand for the Italian market for the years 2025-2029.

## Purpose

The primary purpose of this notebook is to:
1. Load and preprocess historical energy demand data
2. Engineer relevant features for time series forecasting
3. Train an XGBoost regression model for demand prediction
4. Evaluate model performance using appropriate metrics
5. Generate future demand forecasts for 2025-2029
6. Analyze the relationship between energy prices and demand

## Model Architecture

### XGBoost Regressor

The model uses XGBoost (eXtreme Gradient Boosting), a powerful ensemble learning algorithm based on gradient boosted decision trees. XGBoost is particularly well-suited for this task due to:

- Strong performance on tabular data
- Ability to capture non-linear relationships
- Built-in regularization to prevent overfitting
- Efficient handling of missing values
- Feature importance calculation

### Hyperparameters

The model is configured with the following hyperparameters:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| n_estimators | 1000 | Maximum number of trees (early stopping prevents using all) |
| learning_rate | 0.05 | Conservative rate to prevent overfitting |
| max_depth | 5 | Moderate tree depth to balance model complexity |
| subsample | 0.8 | Random sampling of 80% of training instances per tree |
| colsample_bytree | 0.8 | Random sampling of 80% of features per tree |
| objective | 'reg:squarederror' | Mean squared error objective function |
| eval_metric | 'rmse' | Root mean squared error for evaluation |
| early_stopping_rounds | 50 | Stop training if no improvement after 50 rounds |
| random_state | 42 | Seed for reproducibility |

## Feature Engineering

The model uses the following features to predict energy demand:

### Time-Based Features
- Year, month, dayofweek, quarter, dayofyear
- Weekend indicator (is_weekend)

### Lagged Features
- Previous day's demand (demand_lag1)
- Demand from 7 days ago (demand_lag7)
- Demand from 30 days ago (demand_lag30)

### Rolling Window Statistics
- 7-day rolling mean and standard deviation
- 30-day rolling mean and standard deviation

## Data Split

Data is split chronologically (not randomly) to ensure proper time series modeling:
- Training data: Up to December 31, 2022
- Testing data: January 1, 2023 onwards

This chronological split ensures that the model is evaluated on its ability to forecast future data points, rather than interpolate between known values.

## Prediction Methodology

The future prediction process follows these steps:

1. Initialize the forecasting timeframe using the price forecast dates (2025-2029)
2. Create time-based features for the future dates
3. Initialize lag features using the most recent historical values
4. Iteratively predict each day in sequence:
   - Predict the current day using available features
   - Update lag features based on the new prediction
   - Update rolling statistics
   - Move to the next day
5. Save the complete forecast to a CSV file

This iterative approach allows the model to use its own predictions as inputs for future predictions, simulating the real-world forecasting process.

## Performance Metrics

The model is evaluated using multiple metrics:

- **RMSE (Root Mean Squared Error)**: Measures the square root of the average squared differences between predicted and actual values. More sensitive to large errors.
- **MAE (Mean Absolute Error)**: Measures the average magnitude of errors without considering direction. More robust to outliers than RMSE.
- **R² (Coefficient of Determination)**: Indicates the proportion of variance in the dependent variable that is predictable from the independent variables.

## Price-Demand Correlation Analysis

The notebook includes an analysis of the correlation between energy prices and demand:

1. Historical correlation: Analyzes the relationship in past data
2. Future correlation: Examines the predicted relationship in future data
3. Visualizations: Includes scatter plots and time series visualizations

Understanding this correlation is crucial for energy planning, as it helps predict how demand might change in response to price fluctuations.

## Usage

To run this notebook:

1. Ensure all dependencies are installed
2. Verify data files are available in the correct locations
3. Run all cells in sequence
4. Review the generated outputs and visualizations

## Outputs

The notebook generates:

- A trained XGBoost model saved as 'energy_demand_xgb_v1.joblib'
- Feature importance visualization
- Actual vs. predicted demand plots
- Price-demand correlation visualizations
- Future demand predictions (CSV file)

## Limitations and Assumptions

- The model assumes that historical patterns will continue into the future
- Extreme events (e.g., pandemics, major policy changes) are not explicitly modeled
- Weather data is not included, which could improve predictions
- Economic indicators are not directly incorporated
- The model does not account for potential structural changes in energy consumption patterns

## Future Improvements

Potential enhancements for future versions:

- Include weather data as features
- Incorporate economic indicators
- Consider ensemble approaches combining multiple models
- Implement more sophisticated time series techniques
- Add confidence intervals to predictions
- Incorporate additional external features that might influence energy demand
- Experiment with deep learning approaches (e.g., LSTM, Transformer models) 