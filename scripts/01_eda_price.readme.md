# Energy Price Data Exploratory Analysis Script

## Overview
This script performs exploratory data analysis (EDA) on the processed energy price data for the Italian energy market from 2015-2024. It includes time series decomposition, stationarity tests, autocorrelation analysis, feature engineering, and visualization generation to understand patterns and prepare data for forecasting.

## Purpose
- Analyze temporal patterns and trends in energy price data
- Decompose the time series into trend, seasonal, and residual components
- Test for stationarity using ADF and KPSS tests
- Examine autocorrelation and partial autocorrelation patterns
- Create time-based features and lag variables for forecasting
- Generate visualizations for pattern identification
- Prepare train/test split for subsequent modeling

## Data Requirements
- Processed price data file (`data/final/Italy/energy_price2015_2024.csv`)
  - This file should contain daily price data with columns:
    - `Date`: Date in YYYY-MM-DD format
    - `price_eur_mwh`: Price in euros per megawatt-hour
    - Other columns from the processing script (is_outlier, year, month, day, day_of_week, is_weekend)

## Analysis Steps
1. **Data Loading**
   - Import processed price data from CSV file
   - Convert date column to datetime format
   - Create a time series indexed version of the dataframe

2. **Feature Engineering**
   - Create time-based features:
     - Day of week (0-6) and day name
     - Month (1-12) and month name
     - Quarter (1-4)
     - Year
     - Day of year (1-366)
     - Weekend indicator (0/1)
   - Generate lag features:
     - Previous day price (lag 1)
     - Previous week price (lag 7)
     - Previous month price (lag 30)
   - Calculate rolling window statistics:
     - 7-day moving average
     - 30-day moving average

3. **Time Series Analysis**
   - Decompose time series into trend, seasonal, and residual components
   - Perform stationarity tests:
     - Augmented Dickey-Fuller (ADF) test
     - Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
   - Calculate and plot autocorrelation function (ACF)
   - Calculate and plot partial autocorrelation function (PACF)

4. **Visualization Generation**
   - Time series plot of price over time
   - Distribution plot with mean and median indicators
   - Monthly boxplots showing seasonal patterns
   - Day of week boxplots showing weekly patterns
   - Yearly trend line plot
   - Correlation heatmap for features
   - Scatter plots for lag relationships

5. **Train/Test Split**
   - Create chronological split (80% train, 20% test)
   - Save train and test datasets to CSV files
   - Visualize the split with a plot

## Outputs
- **Visualizations** (saved to `outputs/images/`):
  - `price_time_series.png`: Time series plot of price data
  - `price_decomposition.png`: Time series decomposition components
  - `price_acf_pacf.png`: Autocorrelation and partial autocorrelation plots
  - `price_distribution.png`: Histogram of price distribution
  - `price_monthly_boxplot.png`: Monthly price patterns
  - `price_day_of_week_boxplot.png`: Day of week price patterns
  - `price_yearly_trend.png`: Yearly average price trend
  - `price_correlation_heatmap.png`: Feature correlation heatmap
  - `price_lag_relationships.png`: Scatter plots of price vs lagged prices
  - `price_train_test_split.png`: Visualization of train/test split

- **Data Files** (saved to `data/final/Italy/`):
  - `price_train_data.csv`: Training dataset with all features
  - `price_test_data.csv`: Testing dataset with all features

## Usage Instructions
1. Ensure the processed price data file (`energy_price2015_2024.csv`) is available in the `data/final/Italy/` directory
2. Run the script: `python 01_eda_price.py`
3. Review the terminal output for stationarity test results and other statistics
4. Examine the generated visualizations in the `outputs/images/` directory

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- pathlib
- warnings

## Key Findings
- The script reveals both seasonal and trend components in the Italian energy price data
- Stationarity tests show mixed results (ADF suggests stationarity, KPSS suggests non-stationarity)
- Strong autocorrelation is present, especially at lag 1 and lag 7
- Monthly patterns show seasonal variation in prices
- Day of week patterns show differences between weekdays and weekends
- Lag features (especially lag 1 and lag 7) show strong correlation with current prices
- The data is successfully prepared for subsequent modeling with appropriate train/test split

## Next Steps
- Use the train/test datasets for developing forecasting models
- Consider differencing or other transformations to address non-stationarity
- Explore additional features that might improve forecasting accuracy
- Implement the price forecasting model in the next script (02_train_price_model.py) 