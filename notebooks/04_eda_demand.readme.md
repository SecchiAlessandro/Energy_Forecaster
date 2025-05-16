# Energy Demand Data EDA Notebook

## Overview
This Jupyter notebook performs exploratory data analysis (EDA) on the processed energy demand data for the Italian energy market from 2015-2024. It provides interactive visualizations and analysis to understand demand patterns, temporal trends, and prepare data for forecasting models.

## Purpose
- Analyze temporal patterns and trends in Italian energy demand data
- Decompose time series into trend, seasonal, and residual components
- Test for stationarity and examine autocorrelation patterns
- Create time-based features and lag variables for forecasting
- Generate visualizations to identify patterns and relationships
- Prepare train/test split for subsequent modeling

## Data Requirements
- Processed demand data file (`data/final/Italy/energy_demand2015_2024.csv`)
  - This file should contain daily demand data with columns:
    - `date`: Date in YYYY-MM-DD format
    - `demand_mw`: Daily energy demand in megawatt hours
    - `is_outlier`: Boolean flag indicating if the data point is an outlier
    - `year`, `month`, `day`: Date components

## Notebook Structure
1. **Data Loading**
   - Loads the processed demand data from CSV
   - Converts date to datetime format and sets as index
   - Renames columns for consistency

2. **Basic Statistics**
   - Calculates and displays summary statistics
   - Checks for missing values
   - Displays first few rows of data

3. **Time Series Analysis**
   - Creates time series plots of daily and monthly demand
   - Performs seasonal decomposition to extract trend, seasonal, and residual components
   - Conducts stationarity tests (ADF and KPSS)
   - Generates autocorrelation (ACF) and partial autocorrelation (PACF) plots

4. **Correlation Analysis**
   - Creates time-based features (day of week, month, quarter, etc.)
   - Calculates correlations between demand and time features
   - Analyzes monthly and daily demand patterns

5. **Feature Engineering**
   - Creates lag features (1-day, 7-day, 30-day)
   - Calculates rolling window statistics (7-day and 30-day moving averages and standard deviations)
   - Generates year-over-year comparison features
   - Analyzes correlations between engineered features

6. **Train/Test Split Preparation**
   - Splits data into training (2015-2022) and testing (2023-2024) sets
   - Saves train/test datasets for modeling
   - Creates visualization of the split

## Outputs
- **Data Files**:
  - `data/final/Italy/demand_train_data.csv`: Training dataset for modeling
  - `data/final/Italy/demand_test_data.csv`: Testing dataset for model evaluation

- **Visualizations**:
  - Time series plot of daily and monthly demand
  - Seasonal decomposition of demand time series
  - Autocorrelation and partial autocorrelation plots
  - Correlation of time features with demand
  - Monthly and daily demand patterns
  - Demand and rolling average features
  - Correlation matrix of engineered features
  - Train/test split visualization

## Usage Instructions
1. Ensure the processed demand data file (`energy_demand2015_2024.csv`) is available in the `data/final/Italy/` directory
2. Open the notebook in Jupyter Lab or Jupyter Notebook
3. Run all cells sequentially
4. Review the generated visualizations and statistical analysis
5. Use the train/test datasets for subsequent modeling tasks

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- pathlib

## Key Findings
- The energy demand data shows strong seasonal patterns, with higher demand in winter and summer months
- The time series is stationary according to the ADF test (p-value < 0.05) but borderline non-stationary according to the KPSS test
- Month and day of week are important predictors of demand patterns
- The data shows strong autocorrelation at lags of 1, 7, and multiples of 7 days, indicating weekly patterns
- The trend component shows a general increase in demand over the years with some fluctuations
- Lag features and rolling statistics show high correlation with the target variable, making them valuable for forecasting

## Next Steps
- Use the engineered features to train forecasting models
- Explore additional external features (e.g., weather data, holidays) to improve forecasting
- Investigate the impact of COVID-19 on energy demand patterns
- Compare different modeling approaches for demand forecasting 