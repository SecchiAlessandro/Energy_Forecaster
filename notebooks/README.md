# Energy Market Forecasting Notebooks

This directory contains Jupyter notebooks for the Italian Energy Market Forecasting project. These notebooks document the data processing, exploratory data analysis, and model training steps for forecasting energy prices, demand, and renewable energy sources (RES) generation for the period 2025-2029.

## Notebook Organization

The notebooks are organized in a sequential workflow, with each notebook building on the outputs of previous steps:

### Data Processing Notebooks

1. **00_process_price_data.ipynb**
   - Processes raw price data (PUN) from 2015-2024
   - Cleans, standardizes, and prepares price data for analysis
   - Identifies and handles outliers
   - Generates basic statistics and visualizations

2. **03_process_demand_data.ipynb**
   - Processes raw energy demand data from 2015-2024
   - Aggregates hourly data to daily averages
   - Handles missing values and identifies outliers
   - Merges with price data for joint analysis
   - Generates statistics and visualizations

3. **06_process_res_data.ipynb** (Upcoming)
   - Will process renewable energy sources (RES) generation data
   - Will clean and standardize RES data
   - Will prepare RES data for analysis and modeling

### Exploratory Data Analysis (EDA) Notebooks

4. **01_eda_price.ipynb**
   - Performs exploratory analysis of energy price data
   - Includes time series decomposition and stationarity tests
   - Analyzes autocorrelation patterns
   - Creates visualizations of price trends and patterns

5. **04_eda_demand.ipynb**
   - Performs exploratory analysis of energy demand data
   - Includes time series decomposition and stationarity tests
   - Analyzes correlations between demand and time features
   - Creates visualizations of demand patterns by month and day of week
   - Engineers features for demand forecasting
   - Prepares train/test split for modeling

6. **07_eda_res_generation.ipynb** (Upcoming)
   - Will perform exploratory analysis of RES generation data
   - Will analyze patterns in renewable energy production
   - Will create visualizations of generation trends

### Model Training Notebooks

7. **02_train_price_model.ipynb**
   - Trains XGBoost models for price forecasting
   - Evaluates model performance with RMSE, MAE, and R²
   - Generates feature importance analysis
   - Creates future price predictions for 2025-2029
   - Produces visualizations for model interpretation

8. **05_train_demand_model.ipynb** (Upcoming)
   - Will train XGBoost models for demand forecasting
   - Will evaluate model performance
   - Will generate future demand predictions

9. **08_train_res_model.ipynb**
   - Trains XGBoost models for RES generation forecasting
   - Evaluates model performance with RMSE, MAE, and R²
   - Generates feature importance analysis
   - Creates future RES generation predictions for 2025-2029
   - Produces visualizations for model interpretation

### Analysis Notebooks

10. **09_demand_supply_analysis.ipynb** (Upcoming)
    - Will analyze the relationship between energy demand and RES generation
    - Will calculate monthly and yearly aggregations
    - Will compute key metrics like coverage and surplus/deficit
    - Will generate visualizations and save results

## Available Notebooks

### 1. RES Generation Model Training (`08_train_res_model.ipynb`)

**Purpose:** Train an XGBoost model to forecast Renewable Energy Sources (RES) generation for Italy for the years 2025-2029.

**Data Requirements:**
- Processed RES generation data from 2015-2024 (`data/final/Italy/res_generation2015_2024.csv`)

**Processing Steps:**
1. Load and prepare RES generation data
   - Convert date to datetime and set as index
   - Create time-based features (year, month, day of week, etc.)
   - Create lag features (1, 7, 14, 30 days)
   - Create rolling window features (7, 14, 30 days)
   - Create seasonal features using sine/cosine transformations

2. Feature Selection
   - Time features (month, day of week, quarter, etc.)
   - Lag features
   - Rolling window statistics
   - Source-specific features (solar, wind, hydro, etc.)

3. Model Training
   - Split data into training (before 2023) and validation (2023-2024)
   - Train XGBoost regression model
   - Evaluate performance with RMSE, MAE, and R²

4. Future Predictions
   - Generate daily RES generation forecasts for 2025-2029
   - Save predictions to `data/final/Italy/res_generation2025_2029.csv`

**Outputs:**
- Trained XGBoost model (`models/res_generation_xgb_v1.joblib`)
- Feature list (`models/res_generation_features.joblib`)
- Future predictions (`data/final/Italy/res_generation2025_2029.csv`)
- Visualizations:
  - Actual vs. Predicted plot (`outputs/images/res_actual_vs_predicted.png`)
  - Feature importance plot (`outputs/images/res_feature_importance.png`)
  - Scatter plot of actual vs. predicted (`outputs/images/res_scatter_actual_vs_predicted.png`)
  - Historical and future predictions plot (`outputs/images/res_future_predictions.png`)

**Key Findings:**
- The model successfully captures seasonal patterns in RES generation
- Feature importance analysis reveals the most influential factors for RES generation
- The model achieves good performance on the validation set
- Future predictions follow expected seasonal patterns while accounting for historical trends

## Running the Notebooks

To run these notebooks:

1. Ensure you have the required dependencies installed:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
   ```

2. Make sure the required data files are available in the specified paths

3. Run the notebook cells in sequence:
   ```
   jupyter notebook notebooks/08_train_res_model.ipynb
   ```

## Note on Path Configuration

The notebooks use relative paths based on the project structure. If you encounter path-related errors, you may need to adjust the `BASE_DIR` variable in the notebook.

## Usage Instructions

Each notebook has its own detailed README file (e.g., `03_process_demand_data.readme.md`) that provides specific information about:

- The purpose and functionality of the notebook
- Input data requirements
- Processing steps
- Output files generated
- Key findings and insights
- Usage instructions
- Dependencies and assumptions

To run these notebooks:

1. Ensure you have activated the project's virtual environment
2. Make sure all required data files are in their expected locations
3. Run the notebooks in the numbered sequence
4. Check the output directories for generated files and visualizations

## Dependencies

All notebooks rely on the project's main dependencies, which are listed in the `requirements.txt` file in the project root directory. The core libraries used include:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- statsmodels

## Output Locations

- Processed data: `data/processed/Italy/`
- Final datasets: `data/final/Italy/`
- Visualizations: `outputs/images/`
- Trained models: `models/`

## Data Sources

The notebooks use data from the following sources:
- Italian energy price data (PUN) from 2015-2024
- Italian energy demand data from 2015-2024
- Italian RES generation data from 2015-2024

All raw data files should be placed in the appropriate subdirectories under `data/raw/` before running the processing notebooks. 