# Energy Price Data Processing Notebook

## Overview
This notebook processes raw PUN (Prezzo Unico Nazionale) energy price data for the Italian energy market, covering the period from 2015 to 2024. It handles data cleaning, preprocessing, and generates visualizations to understand price patterns and trends.

## Purpose
- Load and clean raw energy price data from the Italian electricity market
- Handle missing values and identify outliers using appropriate methods
- Create time-based features for analysis (year, month, day, weekday)
- Generate statistical summaries and visualizations of price patterns
- Save processed data for downstream analysis and modeling

## Data Requirements
- Raw PUN price data file (`data/raw/energy_price/Italy/PUN.csv`)
  - This file should contain daily price data with columns:
    - `Date`: Date in DD/MM/YYYY format
    - `€/MWh`: Price in euros per megawatt-hour

## Processing Steps
1. **Data Loading**
   - Import raw price data from CSV file
   - Validate data structure and availability

2. **Data Cleaning and Preprocessing**
   - Standardize column names
   - Convert date format from DD/MM/YYYY to YYYY-MM-DD
   - Set date as index and sort chronologically
   - Handle missing values using time-series appropriate methods:
     - Forward fill (with 3-day window limit)
     - Backward fill (if needed)
     - Median imputation (for any remaining missing values)
   - Identify outliers using the IQR method (Q1 - 3*IQR, Q3 + 3*IQR)
   - Add time-based features:
     - Year, month, day
     - Day of week
     - Weekend flag

3. **Final Dataset Creation**
   - Filter data to include only 2015-2024 period
   - Reset index to make date a column
   - Convert date to string format for compatibility

4. **Statistical Analysis**
   - Calculate summary statistics (count, mean, std, min, percentiles, max)
   - Compute yearly price averages
   - Compute monthly price averages across all years
   - Compare weekday vs. weekend price patterns

5. **Visualization Generation**
   - Time series plot of price trends
   - Yearly box plots showing price distribution by year
   - Monthly box plots showing seasonal patterns
   - Overall price distribution histogram
   - Outlier visualization

## Outputs
- **Processed Data**
  - Intermediate processed file: `data/processed/Italy/PUN_p.csv`
  - Final dataset: `data/final/Italy/energy_price2015_2024.csv`

- **Visualizations**
  - Time series plot: `outputs/images/price_timeseries.png`
  - Yearly distribution: `outputs/images/yearly_price_distribution.png`
  - Monthly distribution: `outputs/images/monthly_price_distribution.png`
  - Price distribution: `outputs/images/price_distribution.png`
  - Outlier visualization: `outputs/images/price_outliers.png`

## Usage Instructions
1. Ensure the raw data file (`PUN.csv`) is available in the `data/raw/energy_price/Italy/` directory
2. Open the notebook in Jupyter
3. Run all cells in the notebook in sequence
4. Review the generated statistics and visualizations displayed inline in the notebook
5. Check the output directories for the processed data files

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- datetime
- pathlib

## Key Findings
- The notebook reveals seasonal patterns in Italian energy prices
- Price volatility varies significantly across different years
- Weekend prices tend to be lower than weekday prices
- Outliers are identified and visualized but preserved in the dataset
- The processed data maintains the temporal structure needed for time series analysis

## Notes
- The notebook includes additional explanatory text and background information about the PUN price metric
- Interactive visualizations allow for deeper exploration of the price patterns
- The notebook uses a 3*IQR threshold for outlier detection, which is more conservative than the typical 1.5*IQR
- Missing values are handled using time-series appropriate methods that preserve temporal patterns
- The notebook creates all necessary directories if they don't exist
- All visualizations are both displayed in the notebook and saved as PNG files with 300 DPI resolution 