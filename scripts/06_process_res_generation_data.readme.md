# Renewable Energy Sources (RES) Generation Data Processing

This notebook processes raw renewable energy sources (RES) generation data for Italy from 2015-2024 and prepares it for forecasting and analysis. The notebook handles data loading, cleaning, aggregation, and visualization of renewable energy generation patterns.

## Purpose

The primary purpose of this notebook is to:
1. Load raw RES generation data from multiple years (2015-2024)
2. Process and clean the data to ensure consistency
3. Create a combined RES generation metric from multiple renewable sources
4. Aggregate hourly data to daily averages
5. Generate statistics and visualizations to understand RES generation patterns
6. Prepare the data for future forecasting models

## Data Sources

The notebook processes data from the following sources:
- Raw RES generation files (`RES_generation20XX.csv`) for years 2015-2024
- Located in `data/raw/res_generation/Italy/` directory
- Each file contains hourly generation data for multiple renewable sources

## Renewable Energy Sources

The notebook combines generation data from multiple renewable sources:
- **Geothermal**: Energy extracted from heat stored in the earth
- **Hydro Pumped Storage (net)**: Energy from pumped hydroelectric storage systems
- **Hydro Run-of-river and poundage**: Energy from flowing river water
- **Hydro Water Reservoir**: Energy from stored water in reservoirs
- **Solar**: Photovoltaic and concentrated solar power
- **Wind Offshore**: Energy from wind turbines located in bodies of water
- **Wind Onshore**: Energy from land-based wind turbines

## Processing Methodology

### Data Loading
- Iteratively loads CSV files for each year using a consistent naming pattern
- Handles missing files gracefully
- Validates data structure and required columns

### Data Cleaning
- Standardizes date/time formats
- Handles missing values through appropriate imputation techniques
- Identifies and addresses outliers
- Ensures consistent units and measurements

### Combined RES Generation
- Creates a `Total_RES` column by summing all renewable sources
- Ensures proper handling of missing values in individual sources
- Provides a comprehensive metric of total renewable generation

### Temporal Aggregation
- Converts hourly data to daily averages
- Maintains proper time series structure
- Preserves temporal patterns important for forecasting

## Visualizations

The notebook includes several visualizations to understand RES generation patterns:

### Time Series Analysis
- **Daily RES Generation**: Shows the daily total renewable energy generation over time
- **Monthly Averages**: Displays seasonal patterns in renewable generation
- **Yearly Trends**: Highlights long-term trends in renewable energy adoption

### Source Contribution Analysis
- **Source Distribution**: Pie charts showing the relative contribution of each renewable source
- **Source Trends**: Line charts showing how each source's contribution has changed over time
- **Seasonal Variations by Source**: Heatmaps showing how different sources perform across seasons

### Statistical Analysis
- **Distribution of Generation Values**: Histograms of daily generation amounts
- **Correlation with Weather Factors**: Scatter plots showing relationships with available weather data
- **Year-over-Year Comparison**: Box plots comparing generation statistics across years

## Output Files

The notebook generates the following output files:

1. **Processed Yearly Files**:
   - Location: `data/processed/Italy/RES_generation20XX_p.csv`
   - Content: Cleaned and aggregated daily generation data for each year
   - Format: CSV with standardized columns

2. **Combined Dataset**:
   - Location: `data/final/Italy/res_generation2015_2024.csv`
   - Content: Concatenated data from all processed yearly files
   - Format: CSV with standardized columns including the combined RES metric

## Usage

To run this notebook:

1. Ensure raw data files are available in the correct directory structure
2. Execute all cells in sequence
3. Review the generated visualizations to understand RES generation patterns
4. Verify that output files are created in the expected locations

## Data Structure

The processed data includes the following key columns:

| Column | Description |
|--------|-------------|
| date | Date in YYYY-MM-DD format |
| Total_RES | Combined daily average generation from all renewable sources (MW) |
| Geothermal | Daily average generation from geothermal sources (MW) |
| Hydro_* | Daily average generation from various hydro sources (MW) |
| Solar | Daily average generation from solar sources (MW) |
| Wind_* | Daily average generation from wind sources (MW) |

## Limitations and Considerations

- The data processing assumes consistent formatting across yearly files
- Some renewable sources may have missing data for certain periods
- Daily aggregation smooths out intraday variations that might be important for some analyses
- Weather data is not incorporated, which could provide context for generation patterns
- The combined RES metric treats all sources equally without considering reliability or dispatchability

## Future Improvements

Potential enhancements for future versions:

- Incorporate weather data to explain generation variations
- Add forecasting capabilities for future RES generation
- Include more sophisticated outlier detection methods
- Add interactive visualizations for exploring patterns
- Implement source-specific cleaning procedures
- Calculate reliability metrics for different renewable sources 