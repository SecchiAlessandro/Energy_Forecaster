# Energy Forecaster

A comprehensive energy market forecasting tool that analyzes and predicts Italian energy prices, demand, and renewable energy generation.

## Project Overview

This project processes historical Italian energy market data to build forecasting models that predict future energy prices, demand, and renewable energy generation. The analysis also includes an assessment of how renewable energy sources might cover future energy demands.

## Features

- **Energy Price Forecasting**: Predicts future energy prices based on historical trends and patterns
- **Energy Demand Forecasting**: Forecasts future energy consumption using XGBoost models
- **Renewable Energy Generation Prediction**: Projects future solar, wind, hydro, and other renewable energy generation
- **Supply-Demand Analysis**: Calculates the percentage of demand that can be covered by renewable sources
- **Comprehensive Visualization**: Generates multiple plots and charts to visualize historical data and future predictions

## Project Structure

```
.
├── data/               # Data directory
│   ├── raw/            # Raw data files
│   ├── processed/      # Processed data files
│   └── final/          # Final analysis results
├── scripts/            # Data processing and analysis scripts
├── models/             # Trained ML models
├── notebooks/          # Jupyter notebooks for exploration
├── outputs/            # Output files and visualizations
│   └── images/         # Generated plots and charts
├── main.py             # Main pipeline execution script
└── requirements.txt    # Python dependencies
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Energy_Forecaster.git
   cd Energy_Forecaster
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main pipeline to execute all data processing and analysis scripts:

```
python main.py
```

This will:
1. Set up the necessary directory structure
2. Process raw energy price, demand, and renewable generation data
3. Perform exploratory data analysis
4. Train forecasting models
5. Generate predictions for the 2025-2029 period
6. Analyze the relationship between energy supply and demand
7. Create visualizations of the results

## Data Sources

The project uses historical Italian energy market data from 2015-2024, including:
- Energy prices (EUR/MWh)
- Energy demand (MW)
- Renewable energy generation by source (solar, wind, hydro, etc.)

## Output

The main outputs include:
- Forecasting models for energy prices, demand, and renewable generation
- CSV files with predictions for 2025-2029
- Visualization plots showing historical trends and future predictions
- Analysis reports on renewable energy coverage of future demand

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Italian energy market data provided by public energy authorities
- This project uses XGBoost for machine learning models 