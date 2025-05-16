# Energy Demand Data Processing Notebook

## Overview

This notebook (`03_process_demand_data.ipynb`) processes raw energy demand data for Italy from 2015 to 2024. It is part of a larger energy market forecasting project focusing on the Italian electricity market.

## Purpose

The notebook performs comprehensive data cleaning, standardization, and aggregation of hourly energy demand data to prepare it for further analysis and modeling. It serves as a critical preprocessing step in the energy forecasting pipeline.

## Input Data Requirements

- **Raw Demand Data**: CSV files containing hourly energy demand data for Italy from 2015 to 2024
  - Location: `data/raw/energy_demand/Italy/consumption20XX.csv` (where XX represents the year)
  - Format: CSV files with hourly timestamps and demand values in MW
  - Required columns: 
    - `Time (CET/CEST)`: Datetime column in format `DD.MM.YYYY HH:MM`
    - `Actual Total Load [MW]`: Energy demand values

- **Price Data** (for merging): Processed price data from the Italian energy market
  - Location: `data/processed/Italy/PUN_p.csv`
  - Required for the demand-price correlation analysis

## Processing Steps

The notebook performs the following operations:

1. **Data Loading**: Imports raw hourly demand data from yearly CSV files (2015-2024)
2. **Standardization**: Normalizes column names and formats across all files
3. **Missing Value Handling**: Uses time-based interpolation to fill gaps in the data
4. **Aggregation**: Converts hourly data to daily average demand values
5. **Outlier Detection**: Identifies outliers using the Interquartile Range (IQR) method
6. **Date Component Extraction**: Adds year, month, and day columns for time-based analysis
7. **Data Merging**: Combines demand data with price data when available
8. **Statistical Analysis**: Generates summary statistics and visualizations

## Output Files Generated

The notebook produces the following outputs:

1. **Processed Yearly Files**: 
   - Location: `data/processed/Italy/consumption20XX_p.csv`
   - Content: Cleaned and aggregated daily demand data for each year

2. **Combined Demand Dataset**:
   - Location: `data/final/Italy/energy_demand2015_2024.csv`
   - Content: All years of processed demand data in a single file

3. **Merged Demand-Price Dataset**:
   - Location: `data/final/Italy/energy_demand2015_2024_merged.csv`
   - Content: Demand data merged with corresponding price data

4. **Visualizations**:
   - Location: `outputs/images/`
   - Files:
     - `demand_timeseries.png`: Time series plot of daily demand
     - `yearly_demand_distribution.png`: Box plots of demand by year
     - `monthly_demand_distribution.png`: Box plots of demand by month
     - `demand_distribution.png`: Histogram of demand distribution
     - `demand_outliers.png`: Scatter plot highlighting outliers
     - `monthly_demand_heatmap.png`: Heatmap of monthly demand by year
     - `demand_trend.png`: Trend analysis with 30-day moving average
     - `demand_price_correlation.png`: Scatter plot of demand vs. price

## Key Findings

The notebook reveals several important insights about Italian energy demand:

1. The average daily energy demand over the period is approximately 32,600 MW
2. There are clear seasonal patterns, with higher demand typically observed in winter months
3. Yearly trends show variations that may correlate with economic factors and weather patterns
4. Several outliers were identified, primarily during extreme weather events
5. When analyzed alongside price data, demand shows a moderate correlation with energy prices

## Usage Instructions

To run this notebook:

1. Ensure all raw data files are in the correct locations
2. Verify that the Python environment has all required dependencies (pandas, numpy, matplotlib, seaborn)
3. Run all cells sequentially from top to bottom
4. Review the generated visualizations and statistics
5. Check the output directories for the processed data files

## Dependencies

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- pathlib

## Notes and Assumptions

- The notebook assumes daily aggregation is appropriate for the analysis goals
- Outliers are identified but not removed, only flagged for further analysis
- The IQR method uses a factor of 3 (rather than the typical 1.5) to be more conservative in outlier detection
- When interpolating missing values, time-based interpolation is used to account for the temporal nature of the data 