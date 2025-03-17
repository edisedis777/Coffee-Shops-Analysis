# Coffee Shops Analysis
This project analyzes the Foursquare Open Source Places dataset to explore the distribution of coffee shops across the United States, with a special focus on Portland, Oregon. It provides insights into top coffee chains, their locations, open versus closed status, and closure trends over time.

### Initial Fetch
<img width="526" alt="Screenshot 2025-03-17 at 10 24 55" src="https://github.com/user-attachments/assets/7c393d66-00e2-4360-a5ce-e5e13693486f" />

### Analysis Pipeline
<img width="419" alt="Screenshot 2025-03-17 at 12 09 04" src="https://github.com/user-attachments/assets/12de2ec5-64ae-42fd-a9f8-8eac696dec16" />

### Output Visualization Dutch Bros Coffee Shops Locations
![dutch-bros-coffee-shops-locations](https://github.com/user-attachments/assets/7b714580-ba02-4aef-b2ff-7b2a0a1b67c3)

### Output Visualization Top Chains Bar Chart

![top_chains_bar_chart](https://github.com/user-attachments/assets/461a5a64-0950-45fd-9f1f-1c1329d02e7d)

The project is divided into two main components:

1. Analysis Script (coffee_shops_analysis.py): Processes the dataset, performs data analysis, and generates summary reports.
2. Visualization Module (coffee-shops-viz.py): Creates interactive visualizations such as bar charts, choropleth maps, and heatmaps based on the analyzed data.

## Components
### 1. Analysis Script (coffee_shops_analysis.py)
This script fetches and processes the Foursquare Places dataset (from S3 or a local file), analyzes coffee shop distribution, and generates summary reports. It supports caching for faster subsequent runs and can create maps for top coffee chains. 

Key features include:
- Filtering and processing coffee shop data (excluding Starbucks).
- Analyzing top chains by location count.
- Examining open versus closed status and closure trends.
- Exporting processed data in CSV, Parquet, and GeoJSON formats.

### 2. Visualization Module (coffee-shops-viz.py)
This module provides functions to create various interactive visualizations based on the analyzed data. It can be used independently with properly formatted data or alongside the analysis script. 

Visualizations include:
- Bar charts of top coffee chains and open/closed status.
- Choropleth maps of coffee shops by state.
- Heatmaps and marker maps for specific chains or regions (e.g., Portland, OR).
- Closure trend charts and comprehensive dashboards.

## Dependencies
To run this project, you need the following Python libraries:

- pandas
- polars
- numpy
- matplotlib
- altair
- plotly
- folium
- requests
- pyarrow
For optimized data processing, you can optionally install Daft.

## Installation
Install the required libraries using pip:

bash
pip install pandas polars numpy matplotlib altair plotly folium requests pyarrow
To include Daft (optional):

bash
pip install getdaft

## Usage
### 1. Analyzing Coffee Shop Data
Run the coffee_shops_analysis.py script to process and analyze the coffee shop data. It accepts several command-line arguments to customize the analysis:

bash
python coffee_shops_analysis.py [--output OUTPUT_DIR] [--save-maps] [--chains TOP_N] [--skip-daft] [--local-data FILE]

### Arguments
- --output OUTPUT_DIR: Directory to save output files (default: ./output).
- --save-maps: Save generated maps as HTML files.
- --chains TOP_N: Number of top chains to analyze (default: 5).
- --skip-daft: Skip Daft processing and load processed CSV if available.
- --local-data FILE: Use a local Parquet/CSV file instead of fetching from S3.

#### Example
To analyze the top 10 coffee chains and save maps:

bash
python coffee_shops_analysis.py --chains 10 --save-maps

#### This command will:
- Fetch and process the dataset (or load from cache/local file).
- Analyze the top 10 coffee chains.
- Generate and save interactive maps for each chain and a US heatmap.
- Save summary reports and processed data in the ./output directory.

### 2. Creating Visualizations
The coffee-shops-viz.py module offers functions to generate visualizations from the analyzed data. 

Import and use it in your Python code as follows:
python
from coffee_shops_viz import CoffeeShopVisualizer

#### Assuming you have analyzed data (e.g., from coffee_shops_analysis.py)
- viz = CoffeeShopVisualizer(output_dir="./output")
- viz.create_chain_bar_chart(top_chains)          # Bar chart of top chains
- viz.create_open_closed_chart(status_df)         # Open vs. closed status chart
- viz.create_portland_map(pdx_df)                 # Portland coffee shop map
#### ... additional visualization methods available
This will generate HTML and PNG files in the specified output directory (./output/visualizations).

## Data Source
The analysis utilizes the Foursquare Open Source Places dataset, offering comprehensive location data for places across the United States.

## Outputs
The project generates the following outputs in the specified output directory (default: ./output):

#### Summary Reports: Text file (coffee_shops_summary.txt) with key statistics.
#### Processed Data:
- us_coffee_shops.csv and us_coffee_shops.parquet (all US coffee shops).
- portland_coffee_shops.csv (Portland-specific data).
- coffee_shops.geojson (for mapping applications).
- Maps (if --save-maps is used):
- HTML files for top chain locations, US heatmap, and Portland map (in ./output/maps).
- Visualizations (from coffee-shops-viz.py):
- HTML and PNG files for charts and maps (in ./output/visualizations).

### Contributing
If you encounter issues or have suggestions for improvements, please open an issue on the GitHub repository.

### Credits
- Daft
- Polars
- Simon Willson's article (https://simonwillison.net/search/?q=Foursquare+OS+Places) 

License
MIT License - feel free to use, modify, and distribute!
