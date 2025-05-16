# Energy Price Data EDA Notebook

## Overview
This Jupyter notebook performs exploratory data analysis (EDA) on the processed energy price data for the Italian energy market from 2015-2024. It provides interactive visualizations and analysis to understand price patterns, temporal trends, and prepare data for forecasting models.

## Purpose
- Analyze temporal patterns and trends in Italian energy price data
- Decompose time series into trend, seasonal, and residual components
- Test for stationarity using ADF and KPSS tests
- Examine autocorrelation and partial autocorrelation patterns
- Create time-based features and lag variables for forecasting
- Generate interactive visualizations to identify patterns and relationships
- Prepare train/test split for subsequent modeling

## Data Requirements
- Processed price data file (`data/final/Italy/energy_price2015_2024.csv`)
  - This file should contain daily price data with columns:
    - `Date`: Date in YYYY-MM-DD format
    - `price_eur_mwh`: Price in euros per megawatt-hour
    - Other columns from the processing script (is_outlier, year, month, day, day_of_week, is_weekend)

## Notebook Structure
1. **Setup**
   - Import required libraries
   - Configure paths and visualization settings

2. **Data Loading**
   - Load processed price data from CSV
   - Convert date column to datetime format
   - Create time series indexed version

3. **Feature Engineering**
   - Create time-based features (day of week, month, quarter, year, etc.)
   - Generate lag features (1-day, 7-day, 30-day)
   - Calculate rolling window statistics (7-day and 30-day moving averages)

4. **Time Series Analysis**
   - Decompose time series (trend, seasonal, residual components)
   - Perform stationarity tests (ADF and KPSS)
   - Calculate and plot ACF and PACF

5. **Visualization Generation**
   - Time series plots
   - Distribution analysis
   - Seasonal patterns (monthly, day of week)
   - Correlation analysis
   - Lag relationship visualization

6. **Train/Test Split**
   - Create chronological split (80% train, 20% test)
   - Save datasets to CSV files
   - Visualize the split

## Outputs
- **Interactive Visualizations** (displayed in the notebook):
  - Time series plots with zoom capabilities
  - Decomposition components
  - Autocorrelation and partial autocorrelation plots
  - Distribution histograms with density curves
  - Seasonal pattern boxplots
  - Correlation heatmaps
  - Lag relationship scatter plots
  - Train/test split visualization

- **Saved Visualizations** (saved to `outputs/images/`):
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
2. Open the notebook in Jupyter Lab or Jupyter Notebook
3. Run all cells sequentially
4. Interact with the visualizations to explore data patterns
5. Review the statistical test results and insights

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- pathlib
- warnings

## Key Findings
- The Italian energy price data shows clear seasonal patterns and trends
- Stationarity tests show mixed results, suggesting potential need for differencing
- Strong autocorrelation exists, especially at lag 1 and lag 7
- Monthly patterns show seasonal variation in prices
- Day of week patterns show differences between weekdays and weekends
- Lag features show strong correlation with current prices
- The data is successfully prepared for modeling with appropriate train/test split

## Next Steps
- Use the train/test datasets for developing forecasting models
- Consider differencing or other transformations to address non-stationarity
- Explore additional features that might improve forecasting accuracy
- Implement price forecasting models in the next notebook (02_train_price_model.ipynb) 