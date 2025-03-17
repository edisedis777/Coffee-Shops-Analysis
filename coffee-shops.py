#!/usr/bin/env python3
"""
Coffee Shops Analysis - Analyze Foursquare Places data for coffee shops in the US

This program processes the Foursquare Open Source Places dataset to analyze
coffee shop distribution across the US with a special focus on Portland, OR.
It creates visualizations for coffee shop chains, their distribution, and status.

Data source: https://location.foursquare.com/resources/blog/products/foursquare-open-source-places-a-new-foundational-dataset-for-the-geospatial-community/

Usage:
    python coffee_shops_analysis.py [--output OUTPUT_DIR] [--save-maps] [--chains TOP_N]

Options:
    --output OUTPUT_DIR    Directory to save output files (default: ./output)
    --save-maps            Save generated maps as HTML files
    --chains TOP_N         Number of top chains to analyze (default: 5)
    --skip-daft            Skip daft processing and load processed CSV if available
    --local-data FILE      Use local parquet/CSV file instead of S3
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Data processing
import numpy as np
import pandas as pd
import polars as pl

# Visualizations
import altair as alt
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
from tqdm import tqdm

# For handling S3 data
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.fs as fs

# Optional: Only import if available
try:
    import daft
    from daft.expressions import col
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    print("Warning: Daft not available. Some functionality will be limited.")


class CoffeeShopAnalyzer:
    """Analyze coffee shop data from Foursquare Places dataset."""
    
    def __init__(self, args):
        """Initialize the analyzer with command-line arguments."""
        self.args = args
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.df_pd = None       # pandas DataFrame
        self.polarsdf = None    # polars DataFrame
        self.total_rows = 0     # Total rows in raw dataset
        self.us_coffee_shops = 0  # Number of coffee shops in US
        self.pdx_coffee_shops = 0  # Number of coffee shops in Portland, OR
        
        # Cache file for processed data
        self.cache_file = self.output_dir / "us_coffee_shops.csv"
        
    def fetch_and_process_data(self):
        """Fetch data from S3 and process it using Daft or load from cache."""
        if not self.args.skip_daft and DAFT_AVAILABLE:
            print("Fetching and processing data with Daft...")
            self._process_with_daft()
        elif self.args.local_data:
            print(f"Loading local data from {self.args.local_data}...")
            self._load_local_data()
        elif self.cache_file.exists():
            print(f"Loading cached data from {self.cache_file}...")
            self._load_from_cache()
        else:
            sys.exit("Error: Daft not available and no cached data found. Use --local-data option.")
            
        # Create polars DataFrame from pandas
        self.polarsdf = pl.from_pandas(self.df_pd)
        
        # Calculate summary stats
        self.us_coffee_shops = len(self.df_pd)
        self.pdx_df = self.df_pd.query('region == "OR" & locality == "Portland"')
        self.pdx_coffee_shops = len(self.pdx_df)
        
        print(f"Total coffee shops in US: {self.us_coffee_shops}")
        print(f"Coffee shops in Portland, OR: {self.pdx_coffee_shops}")
            
    def _process_with_daft(self):
        """Process the data using Daft for efficient distributed computation."""
        # Initialize daft
        daft.context.set_runner_native()
        
        # Scan data from S3
        path = "s3://fsq-os-places-us-east-1/release/dt=2024-11-19/places/**"
        IO_CONFIG = daft.io.IOConfig(s3=daft.io.S3Config(anonymous=True))
        df = daft.read_parquet(
            path, 
            io_config=IO_CONFIG,
            schema={
                'date_created': daft.DataType.date(),
                'date_refreshed': daft.DataType.date(),
                'date_closed': daft.DataType.date()
            }
        )
        
        # Get total count of rows in dataset
        self.total_rows = df.count().collect().to_pandas()['count'].max()
        print(f"Total places in dataset: {self.total_rows}")
        
        # Define UDF for title case conversion
        @daft.udf(return_dtype=daft.DataType.string())
        def to_title_case(x: daft.Series) -> list:
            values = x.to_pylist()
            return [v.title() if v is not None else None for v in values]
        
        # Filter and transform the data
        us_coffee_df = df.where(
            (col("country") == "US") & # US only
            ((col("name").str.lower().str.contains("coffee")) | 
            (col("name").str.lower().str.contains("roaster"))) & # some roasters also serve coffee
            (~col("name").str.lower().str.contains("starbucks"))  # Starbucks excluded
        ).with_column("name", 
            col("name").str.replace(" and ", " & ")
                      .str.replace(" AND ", " & ")
                      .str.replace(" And ", " & ")
                      .str.replace("'", "")
        ).with_column("name", to_title_case(col("name")))
        
        # Select needed columns and convert to pandas
        dfarrow = us_coffee_df.select(
            "name", "latitude", "longitude", "locality", "region", 
            "postcode", "address", "date_closed"
        ).to_arrow()
        
        # Fix: Convert Arrow table to pandas DataFrame correctly
        self.df_pd = dfarrow.to_pandas()
        self.df_pd['date_closed'] = pd.to_datetime(self.df_pd['date_closed'])
        
        # Save to cache
        self.df_pd.to_csv(self.cache_file, index=False)
        print(f"Saved processed data to {self.cache_file}")
    
    def _load_local_data(self):
        """Load data from a local file."""
        file_path = self.args.local_data
        if file_path.endswith('.csv'):
            self.df_pd = pd.read_csv(file_path)
            self.df_pd['date_closed'] = pd.to_datetime(self.df_pd['date_closed'])
        elif file_path.endswith('.parquet'):
            self.df_pd = pd.read_parquet(file_path)
            self.df_pd['date_closed'] = pd.to_datetime(self.df_pd['date_closed'])
        else:
            sys.exit(f"Unsupported file format: {file_path}")
    
    def _load_from_cache(self):
        """Load preprocessed data from cache."""
        self.df_pd = pd.read_csv(self.cache_file)
        self.df_pd['date_closed'] = pd.to_datetime(self.df_pd['date_closed'])
    
    def analyze_top_chains(self):
        """Analyze top coffee shop chains."""
        # Get the top N coffee shop chains by count
        top_n = self.args.chains
        top_chains = self.df_pd.groupby('name').size().nlargest(top_n)
        
        print(f"\nTop {top_n} Coffee Chains by Location Count:")
        print("-" * 40)
        
        for name, count in top_chains.items():
            total_locs = len(self.df_pd[self.df_pd['name'] == name])
            valid_locs = len(self.df_pd[self.df_pd['name'] == name].dropna(subset=['latitude', 'longitude']))
            regions = self.df_pd[self.df_pd['name'] == name]['region'].nunique()
            
            print(f"{name}:")
            print(f"  - Total Locations: {total_locs}")
            print(f"  - Mapped Locations: {valid_locs}")
            print(f"  - States Present: {regions}")
        
        return top_chains
    
    def create_chain_maps(self, top_chains):
        """Create maps for top coffee chains."""
        maps_dir = self.output_dir / "maps"
        if self.args.save_maps:
            maps_dir.mkdir(exist_ok=True)
        
        for chain_name, _ in top_chains.items():
            print(f"\nCreating map for {chain_name}...")
            
            # Filter data
            chain_locations = self.df_pd[self.df_pd['name'] == chain_name].copy()
            chain_locations = chain_locations.dropna(subset=['latitude', 'longitude'])
            
            # Calculate center of locations 
            center_lat = chain_locations['latitude'].mean()
            center_lon = chain_locations['longitude'].mean()
            
            # Create base map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add markers for all locations
            for _, location in chain_locations.iterrows():
                # Create popup content
                popup_content = f"""
                <div style='width: 200px'>
                    <b>{location['name']}</b><br>
                    Address: {location.get('address', 'N/A')}<br>
                    {location.get('locality', 'N/A')}, {location.get('region', 'N/A')} {location.get('postcode', 'N/A')}
                </div>
                """
                
                # Add marker
                folium.Marker(
                    location=[float(location['latitude']), float(location['longitude'])],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='brown', icon='info-sign')
                ).add_to(m)
        
            # Save map if requested
            if self.args.save_maps:
                map_file = maps_dir / f"{chain_name.replace(' ', '_').lower()}_locations.html"
                m.save(str(map_file))
                print(f"Saved map to {map_file}")
    
    def analyze_open_vs_closed(self, top_chains):
        """Analyze open vs closed status for top chains."""
        print("\nAnalyzing open vs closed status for top chains...")
        
        # Get names of top chains
        top_chain_names = list(top_chains.index)
        
        # Using Polars for efficient analysis
        status_data = (
            self.polarsdf
            .filter(pl.col('name').is_in(top_chain_names))
            .select(['name', 'date_closed'])
            .with_columns(
                status=pl.when(pl.col('date_closed').is_null())
                      .then(pl.lit('Open'))
                      .otherwise(pl.lit('Closed'))
            )
            .group_by(['name', 'status'])
            .agg(pl.len().alias('count'))
            .sort('name')
        )
        
        # Convert to pandas for compatibility with plotting functions
        status_df = status_data.to_pandas()
        
        # Print summary
        for name in top_chain_names:
            chain_data = status_df[status_df['name'] == name]
            total = chain_data['count'].sum()
            open_count = chain_data[chain_data['status'] == 'Open']['count'].sum() if 'Open' in chain_data['status'].values else 0
            closed_count = chain_data[chain_data['status'] == 'Closed']['count'].sum() if 'Closed' in chain_data['status'].values else 0
            open_pct = (open_count / total) * 100 if total > 0 else 0
            
            print(f"{name}:")
            print(f"  - Open: {open_count} ({open_pct:.1f}%)")
            print(f"  - Closed: {closed_count} ({100-open_pct:.1f}%)")
            print(f"  - Total: {total}")
        
        return status_df
    
    def create_us_heat_map(self):
        """Create heat map of coffee shops across the US."""
        print("\nCreating US heat map...")
        
        # Prepare heat map data
        heat_data = [
            [row['latitude'], row['longitude']] 
            for _, row in self.df_pd.iterrows() 
            if pd.notna(row['latitude']) and pd.notna(row['longitude'])
        ]
        
        # Create base map
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='CartoDB positron')
        
        # Create heatmap layer
        heatmap_layer = HeatMap(
            heat_data,
            radius=5,
            blur=5,
            max_zoom=1,
            gradient={
                '0.2': '#fff5c6',
                '0.4': '#ffd780',
                '0.6': '#ff9b57',
                '0.8': '#da4c5f',
                '1.0': '#5c1a33'
            },
            name='Heat Map'
        )
        heatmap_layer.add_to(m)
        
        # Add layer control
        folium.LayerControl(position='topright').add_to(m)
        
        # Save map if requested
        if self.args.save_maps:
            map_file = self.output_dir / "maps" / "us_coffee_heatmap.html"
            m.save(str(map_file))
            print(f"Saved US heat map to {map_file}")
    
    def create_portland_map(self):
        """Create map of coffee shops in Portland, OR."""
        print("\nCreating Portland coffee shop map...")
        
        # Create map centered on Portland
        m = folium.Map(location=[45.52, -122.67], zoom_start=13)
        
        # Add markers for Portland coffee shops
        for _, row in self.pdx_df.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"{row['name']}<br>{row['address']}",
                    icon=folium.Icon(color='brown', icon='coffee', prefix='fa')
                ).add_to(m)
        
        # Save map if requested
        if self.args.save_maps:
            map_file = self.output_dir / "maps" / "portland_coffee_shops.html"
            m.save(str(map_file))
            print(f"Saved Portland map to {map_file}")
    
    def analyze_closures(self):
        """Analyze coffee shop closures over time."""
        print("\nAnalyzing coffee shop closures over time...")
        
        # Filter for shops that have closed since 2020
        df_pd = self.df_pd.copy()
        df_pd['date_closed'] = pd.to_datetime(df_pd['date_closed'], errors='coerce')
        df_recent = df_pd[df_pd['date_closed'].dt.year > 2020].copy()
        closed_shops = df_recent[df_recent['date_closed'].notna()].copy()
        
        # Skip if no closures found
        if len(closed_shops) == 0:
            print("No closure data found for recent years.")
            return None
        
        # Format date for grouping
        closed_shops['year_month'] = closed_shops['date_closed'].dt.to_period('M')
        
        # Get top states by closure count
        top_10_states = closed_shops['region'].value_counts().head(10).index
        state_data = closed_shops[closed_shops['region'].isin(top_10_states)].copy()
        
        # Group by year_month, region, and name
        monthly_chain_data = state_data.groupby(['year_month', 'region', 'name']).size().reset_index(name='chain_closures')
        monthly_chain_data = monthly_chain_data.sort_values('chain_closures', ascending=False)
        
        # Function to get top chains for each state/month
        def get_top_chains(group):
            chain_info = group.groupby('name')['chain_closures'].sum().sort_values(ascending=False).head(5)
            return '<br>'.join([f"{name}: {count} locations" for name, count in chain_info.items()])
        
        # Generate monthly closures summary
        monthly_closures = monthly_chain_data.groupby(['year_month', 'region'])['chain_closures'].sum().reset_index()
        chain_details = monthly_chain_data.groupby(['year_month', 'region']).apply(get_top_chains).reset_index()
        chain_details.columns = ['year_month', 'region', 'chain_details']
        monthly_closures = monthly_closures.merge(chain_details, on=['year_month', 'region'])
        monthly_closures['year_month'] = pd.to_datetime(monthly_closures['year_month'].astype(str))
        
        # Print summary
        print("\nCoffee shop closures by state (top 10):")
        for state in top_10_states:
            state_closures = monthly_closures[monthly_closures['region'] == state]['chain_closures'].sum()
            print(f"{state}: {state_closures} closures")
        
        return monthly_closures
    
    def generate_reports(self):
        """Generate summary reports of the analysis."""
        report_file = self.output_dir / "coffee_shops_summary.txt"
        print(f"\nGenerating summary report to {report_file}...")
        
        with open(report_file, 'w') as f:
            f.write("COFFEE SHOPS ANALYSIS SUMMARY\n")
            f.write("============================\n\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-----------------\n")
            f.write(f"Total coffee shops in US: {self.us_coffee_shops}\n")
            f.write(f"Coffee shops in Portland, OR: {self.pdx_coffee_shops}\n\n")
            
            f.write("TOP CHAINS BY LOCATION COUNT\n")
            f.write("---------------------------\n")
            top_chains = self.df_pd.groupby('name').size().nlargest(self.args.chains)
            for name, count in top_chains.items():
                f.write(f"{name}: {count} locations\n")
            
            f.write("\nSTATE DISTRIBUTION\n")
            f.write("------------------\n")
            state_counts = self.df_pd['region'].value_counts().head(10)
            for state, count in state_counts.items():
                f.write(f"{state}: {count} coffee shops\n")
            
            f.write("\nFor more detailed analysis, see the generated visualizations and maps.\n")
        
        print(f"Summary report generated.")
    
    def export_data(self):
        """Export processed data in various formats."""
        print("\nExporting processed data...")
        
        # Export CSV and parquet
        self.df_pd.to_csv(self.output_dir / "us_coffee_shops.csv", index=False)
        self.df_pd.to_parquet(self.output_dir / "us_coffee_shops.parquet", index=False)
        
        # Export Portland data
        self.pdx_df.to_csv(self.output_dir / "portland_coffee_shops.csv", index=False)
        
        # Export geojson for mapping applications
        geo_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for _, row in self.df_pd.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row['longitude']), float(row['latitude'])]
                    },
                    "properties": {
                        "name": row['name'],
                        "address": row.get('address', ''),
                        "locality": row.get('locality', ''),
                        "region": row.get('region', ''),
                        "postcode": row.get('postcode', ''),
                        "status": "Closed" if pd.notna(row['date_closed']) else "Open"
                    }
                }
                geo_data["features"].append(feature)
        
        with open(self.output_dir / "coffee_shops.geojson", 'w') as f:
            json.dump(geo_data, f)
        
        print(f"Data exported to {self.output_dir}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("\n=== COFFEE SHOPS ANALYSIS ===\n")
        
        # Fetch and process data
        self.fetch_and_process_data()
        
        # Use tqdm for progress tracking
        print("\nRunning analysis pipeline...")
        steps = ["Analyzing top chains", "Creating maps", "Analyzing open vs closed status", 
                "Creating heat maps", "Analyzing closures", "Generating reports", "Exporting data"]
        
        with tqdm(total=len(steps), desc="Analysis Progress") as pbar:
            # Analyze top chains
            pbar.set_description(f"Analysis Progress - {steps[0]}")
            top_chains = self.analyze_top_chains()
            pbar.update(1)
            
            # Create maps for top chains
            pbar.set_description(f"Analysis Progress - {steps[1]}")
            if self.args.save_maps:
                self.create_chain_maps(top_chains)
            pbar.update(1)
            
            # Analyze open vs closed status
            pbar.set_description(f"Analysis Progress - {steps[2]}")
            status_df = self.analyze_open_vs_closed(top_chains)
            pbar.update(1)
            
            # Create US heat map
            pbar.set_description(f"Analysis Progress - {steps[3]}")
            if self.args.save_maps:
                self.create_us_heat_map()
                self.create_portland_map()
            pbar.update(1)
            
            # Analyze closures over time
            pbar.set_description(f"Analysis Progress - {steps[4]}")
            monthly_closures = self.analyze_closures()
            pbar.update(1)
            
            # Generate reports
            pbar.set_description(f"Analysis Progress - {steps[5]}")
            self.generate_reports()
            pbar.update(1)
            
            # Export data
            pbar.set_description(f"Analysis Progress - {steps[6]}")
            self.export_data()
            pbar.update(1)
        
        print("\nAnalysis complete! Results saved to:", self.output_dir)
        
        return {
            "top_chains": top_chains,
            "status_df": status_df,
            "monthly_closures": monthly_closures
        }

def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze coffee shop data from Foursquare Places dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--output", default="./output", 
                        help="Directory to save output files")
    parser.add_argument("--save-maps", action="store_true",
                        help="Save generated maps as HTML files")
    parser.add_argument("--chains", type=int, default=5,
                        help="Number of top chains to analyze")
    parser.add_argument("--skip-daft", action="store_true",
                        help="Skip daft processing and load processed CSV if available")
    parser.add_argument("--local-data", 
                        help="Use local parquet/CSV file instead of S3")
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = CoffeeShopAnalyzer(args)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()