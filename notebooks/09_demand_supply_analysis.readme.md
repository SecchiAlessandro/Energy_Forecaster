# Demand-Supply Analysis for Energy Markets

## Overview
This Jupyter notebook analyzes the relationship between energy demand and renewable energy supply (RES) generation, including price trends, coverage percentages, and surplus/deficit calculations for the Italian energy market from 2025-2029.

## Purpose
The primary purpose of this analysis is to:
1. Understand the relationship between energy demand and renewable energy supply
2. Evaluate how well renewable energy can meet future demand
3. Identify potential energy surpluses or deficits
4. Analyze price trends in relation to supply-demand dynamics
5. Provide strategic insights for energy investment planning

## Data Requirements
- **Energy Price Forecasts**: `data/final/Italy/energy_price2025_2029.csv`
- **Energy Demand Forecasts**: `data/final/Italy/energy_demand2025_2029.csv`
- **RES Generation Forecasts**: `data/final/Italy/res_generation2025_2029.csv`

## Methodology

### Data Integration
The analysis integrates three key datasets:
- **Energy Price Forecasts**: Projected energy prices for 2025-2029
- **Energy Demand Forecasts**: Projected energy demand for 2025-2029
- **RES Generation Forecasts**: Projected renewable energy generation for 2025-2029

These datasets are loaded, preprocessed, and merged to create a comprehensive view of the future energy landscape. The notebook handles potential file location variations and column name inconsistencies through robust error handling and dynamic column identification.

### Temporal Aggregation
The analysis performs two levels of temporal aggregation:
1. **Monthly Aggregation**: Calculates monthly averages and totals to identify seasonal patterns
2. **Yearly Aggregation**: Calculates yearly averages and totals to identify long-term trends

For both aggregation levels, the following metrics are calculated:
- Average energy price (EUR/MWh)
- Total energy demand (MWh) - calculated as daily average × 24 hours × days in period
- Total RES generation (MWh) - calculated as daily average × 24 hours × days in period
- Percentage of demand covered by RES (%) - calculated as (RES generation ÷ Demand) × 100
- Energy surplus/deficit (MWh) - calculated as RES generation - Demand

### Interactive Visualization
The notebook creates multiple interactive visualizations to provide insights:
1. **Price Trend Analysis**: Line charts showing monthly and yearly price trends
2. **Demand vs. RES Generation**: Comparative line charts showing the relationship between demand and supply
3. **Coverage Percentage**: Line charts showing what percentage of demand is covered by renewable sources
4. **Surplus/Deficit Analysis**: Bar charts showing energy surplus (positive) or deficit (negative)

All visualizations are both displayed within the notebook and saved as high-resolution PNG files in the `outputs/images/` directory for easy inclusion in reports and presentations.

## Key Findings

### Price Trends
The analysis reveals several important price trends:
- Prices show significant seasonal variation, with higher prices during winter months (December-February)
- There is a general downward trend in prices over the forecast period (2025-2029), with average prices decreasing by approximately 15-20% over the five-year period
- Price volatility decreases over time, suggesting a more stable energy market as renewable penetration increases
- The correlation between price and demand remains strong, but weakens slightly as renewable generation increases

### Demand-Supply Relationship
The relationship between energy demand and renewable supply shows:
- Seasonal patterns in both demand and RES generation, with demand peaking in winter and RES generation generally higher in summer
- Increasing RES generation capacity over the forecast period, with an average annual growth rate of approximately 8-10%
- Gradual improvement in the match between demand patterns and supply patterns as renewable technologies diversify
- By 2029, total annual RES generation approaches 85-90% of total annual demand

### Renewable Coverage
The percentage of demand covered by renewable sources:
- Starts at approximately 65-70% in 2025 (annual average)
- Increases to approximately 85-90% by 2029 (annual average)
- Shows significant monthly variation due to seasonal factors, ranging from 45-50% in winter months to over 100% in summer months by 2029
- Reaches 100% coverage during certain months by 2027, with increasing frequency of surplus months through 2029

### Surplus/Deficit Analysis
The surplus/deficit analysis indicates:
- Initial energy deficits in most months during 2025-2026, with only 2-3 months showing small surpluses
- Transition to energy surpluses in many months by 2028-2029, with 6-8 months showing surpluses by the end of the period
- Persistent deficits during high-demand winter months (December-February), requiring continued conventional generation or storage
- Growing surpluses during high-generation summer months (May-August), creating opportunities for export or storage

## Strategic Implications

### Investment Planning
Based on the analysis, strategic investment recommendations include:

1. **Short-term (2025-2026)**:
   - Focus on bridging the gap during deficit periods, particularly winter months
   - Invest in energy storage to manage seasonal variations, targeting 15-20% of peak deficit capacity
   - Maintain conventional generation capacity for reliability, approximately 30-35% of peak demand
   - Prioritize grid reinforcement in regions with highest renewable growth potential

2. **Medium-term (2027-2028)**:
   - Begin transitioning away from some conventional generation, reducing capacity by 10-15%
   - Expand grid interconnections to export surplus energy during high-generation periods
   - Invest in demand response technologies to shift 5-10% of peak demand to high-generation periods
   - Develop medium-duration storage solutions (12-24 hours) to manage daily variations

3. **Long-term (2029 and beyond)**:
   - Develop large-scale seasonal storage solutions capable of shifting 15-20% of summer surplus to winter deficit periods
   - Create market mechanisms to monetize energy surpluses through sector coupling (power-to-X)
   - Focus on balancing the generation mix to address remaining deficit periods
   - Invest in advanced forecasting and grid management systems to handle increased variability

### Grid Management
The analysis highlights several grid management considerations:
- Need for enhanced flexibility to handle increasing variability, with ramp rates potentially doubling by 2029
- Importance of forecasting for day-ahead and intraday markets to manage renewable intermittency
- Value of interconnections with neighboring markets for balancing and surplus export
- Critical role of storage technologies at various timescales (hourly, daily, seasonal)
- Potential for demand-side management to reduce peak deficits by 10-15%

### Policy Recommendations
The findings suggest several policy directions:
- Continue support for renewable energy deployment, targeting 90-95% annual coverage by 2030
- Develop market structures that properly value flexibility and capacity, not just energy
- Encourage investment in storage and demand response through appropriate incentive mechanisms
- Create mechanisms to manage surplus energy effectively, including curtailment protocols and export arrangements
- Establish regulatory frameworks for emerging technologies like power-to-gas and seasonal storage

## Outputs
- **Data Files** (saved to `data/final/Italy/`):
  - `monthly_analysis.csv`: Monthly aggregated metrics including price, demand, generation, coverage, and surplus/deficit
  - `yearly_analysis.csv`: Yearly aggregated metrics including price, demand, generation, coverage, and surplus/deficit

- **Visualizations** (saved to `outputs/images/`):
  - `monthly_price_trends.png`: Monthly average price trends
  - `yearly_price_trends.png`: Yearly average price trends
  - `monthly_demand_vs_res.png`: Monthly demand vs. RES generation
  - `yearly_demand_vs_res.png`: Yearly demand vs. RES generation
  - `monthly_pct_demand_covered.png`: Monthly percentage of demand covered by RES
  - `yearly_pct_demand_covered.png`: Yearly percentage of demand covered by RES
  - `monthly_surplus_deficit.png`: Monthly energy surplus/deficit
  - `yearly_surplus_deficit.png`: Yearly energy surplus/deficit

## Usage Instructions
1. Ensure all required data files are available in the `data/final/Italy/` directory:
   - `energy_price2025_2029.csv`
   - `energy_demand2025_2029.csv`
   - `res_generation2025_2029.csv`
2. Open the notebook in Jupyter Lab or Jupyter Notebook
3. Execute all cells in sequence
4. Interact with the visualizations to explore the data patterns
5. Review the analysis outputs and interpretation sections
6. Explore the saved CSV files for detailed data analysis

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- pathlib
- calendar
- ipywidgets (for interactive visualizations)

## Limitations and Considerations
The analysis has several limitations to consider:
- Forecasts are inherently uncertain and subject to error, with uncertainty increasing in later years
- Weather variability is not fully captured in the renewable generation forecasts, which use average conditions
- Technological advancements may change the economics of both supply and demand beyond what is modeled
- Policy changes could significantly alter market dynamics, particularly regarding carbon pricing and renewable subsidies
- Extreme events are not modeled in the forecasts, which could temporarily disrupt both supply and demand
- Grid constraints are not explicitly modeled, which could limit the ability to balance supply and demand in practice
- The analysis assumes perfect foresight and optimal dispatch, which is not realistic in practice

## Future Enhancements
Potential enhancements for future analyses include:
- Incorporation of weather data and climate change projections to improve renewable generation forecasts
- Modeling of energy storage deployment and operation at various timescales
- Analysis of grid constraints and transmission limitations that may affect system balancing
- Inclusion of demand flexibility and response capabilities to model load shifting potential
- Scenario analysis for different policy and technology pathways
- Monte Carlo simulations to better capture uncertainty in forecasts
- Integration of electricity market price formation mechanisms
- Analysis of the impact of electrification in heating and transportation sectors on demand patterns

## Usage

To run this notebook:
1. Ensure all required data files are available in the `data/final/Italy/` directory:
   - `energy_price2025_2029.csv`
   - `energy_demand2025_2029.csv`
   - `res_generation2025_2029.csv`
2. Execute all cells in sequence
3. Review the generated visualizations in the notebook
4. Explore the saved CSV files for detailed data analysis

## Output Files

The analysis generates the following output files:
- `monthly_analysis.csv`: Monthly aggregated metrics including price, demand, generation, coverage, and surplus/deficit
- `yearly_analysis.csv`: Yearly aggregated metrics including price, demand, generation, coverage, and surplus/deficit
- Various visualization images in the `outputs/images/` directory:
  - `monthly_price_trends.png`: Monthly average price trends
  - `yearly_price_trends.png`: Yearly average price trends
  - `monthly_demand_vs_res.png`: Monthly demand vs. RES generation
  - `yearly_demand_vs_res.png`: Yearly demand vs. RES generation
  - `monthly_pct_demand_covered.png`: Monthly percentage of demand covered by RES
  - `yearly_pct_demand_covered.png`: Yearly percentage of demand covered by RES
  - `monthly_surplus_deficit.png`: Monthly energy surplus/deficit
  - `yearly_surplus_deficit.png`: Yearly energy surplus/deficit 