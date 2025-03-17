#!/usr/bin/env python3
"""
Coffee Shop Visualization Module

This module provides functions to create visualizations for the coffee shop analysis.
It's designed to work with the CoffeeShopAnalyzer class but can be used independently
with properly formatted data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
import requests


class CoffeeShopVisualizer:
    """Class for creating visualizations from coffee shop data."""
    
    def __init__(self, output_dir="./output"):
        """Initialize the visualizer with an output directory."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a directory for saving the visualizations
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def create_chain_bar_chart(self, top_chains, title="Top Coffee Chains"):
        """Create a bar chart of top coffee chains by location count."""
        df = pd.DataFrame({
            'Chain': top_chains.index,
            'Count': top_chains.values
        })
        
        fig = px.bar(
            df, 
            x='Chain', 
            y='Count', 
            title=title,
            labels={'Chain': 'Coffee Chain', 'Count': 'Number of Locations'},
            color='Count',
            color_continuous_scale='brwnyl'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='white',
            yaxis_title="Number of Locations",
            xaxis_title="Coffee Chain"
        )
        
        # Save the figure
        fig.write_html(os.path.join(self.viz_dir, 'top_chains_bar_chart.html'))
        fig.write_image(os.path.join(self.viz_dir, 'top_chains_bar_chart.png'))
        
        return fig
    
    def create_open_closed_chart(self, status_df):
        """Create a chart showing open vs closed locations for each chain."""
        # Create an Altair chart
        chart = alt.Chart(status_df).mark_bar().encode(
            x=alt.X('name:N', title='Coffee Chain'),
            y=alt.Y('count:Q', title='Number of Locations'),
            color=alt.Color('status:N', 
                          scale=alt.Scale(domain=['Open', 'Closed'],
                                        range=['#4A2C2A', '#D4B59D'])),
            tooltip=[
                alt.Tooltip('name:N', title='Chain'),
                alt.Tooltip('status:N', title='Status'),
                alt.Tooltip('count:Q', title='Count')
            ]
        ).properties(
            title='Open vs Closed Locations by Coffee Chain',
            width=600,
            height=400
        ).configure_axis(
            labelAngle=45
        )
        
        # Save the chart
        chart.save(os.path.join(self.viz_dir, 'open_closed_chart.html'))
        
        return chart
    
    def create_state_choropleth(self, df):
        """Create a choropleth map of coffee shops by US state."""
        # Calculate coffee shops per state
        state_counts = df['region'].value_counts().reset_index()
        state_counts.columns = ['state', 'count']
        
        # Create a Plotly choropleth map
        fig = px.choropleth(
            state_counts,
            locations='state',
            locationmode='USA-states',
            color='count',
            scope='usa',
            color_continuous_scale='YlOrBr',
            title='Coffee Shops by State',
            labels={'count': 'Number of Coffee Shops'}
        )
        
        fig.update_layout(
            geo=dict(lakecolor='LightBlue'),
            coloraxis_colorbar=dict(title='Number of Coffee Shops')
        )
        
        # Save the figure
        fig.write_html(os.path.join(self.viz_dir, 'state_choropleth.html'))
        fig.write_image(os.path.join(self.viz_dir, 'state_choropleth.png'))
        
        return fig
    
    def create_folium_state_choropleth(self, df):
        """Create a Folium choropleth map of coffee shops by US state."""
        # Calculate coffee shops per state
        state_counts = df['region'].value_counts().to_dict()
        
        # Get US state GeoJSON
        url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"
        try:
            us_states = requests.get(url).json()
        except requests.RequestException:
            print("Warning: Could not fetch US states GeoJSON. Using local data if available.")
            # Try to use local copy if available
            try:
                import json
                with open(os.path.join(self.output_dir, 'us-states.json'), 'r') as f:
                    us_states = json.load(f)
            except:
                print("Error: Could not load US states GeoJSON.")
                return None
        
        # Create base map
        m = folium.Map(
            location=[39.8283, -98.5795],
            zoom_start=4,
            tiles='cartodbpositron'
        )
        
        # Add choropleth layer
        folium.Choropleth(
            geo_data=us_states,
            name='Coffee Shops by State',
            data=state_counts,
            columns=['State', 'Count'],
            key_on='feature.id',
            fill_color='BuGn', 
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Number of Coffee Shops',
            highlight=True
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save the map
        map_file = os.path.join(self.viz_dir, 'state_choropleth_folium.html')
        m.save(map_file)
        
        return m
    
    def create_chain_map(self, df, chain_name):
        """Create a map for a specific coffee chain."""
        # Filter data for the specified chain
        chain_df = df[df['name'] == chain_name].copy()
        chain_df = chain_df.dropna(subset=['latitude', 'longitude'])
        
        if len(chain_df) == 0:
            print(f"No mapping data available for {chain_name}")
            return None
        
        # Calculate center coordinates
        center_lat = chain_df['latitude'].mean()
        center_lon = chain_df['longitude'].mean()
        
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles='cartodbpositron')
        
        # Create marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for chain locations
        for _, row in chain_df.iterrows():
            popup_content = f"""
            <div style="width:200px">
                <b>{row['name']}</b><br>
                {row.get('address', 'No address')}<br>
                {row.get('locality', '')}, {row.get('region', '')} {row.get('postcode', '')}
            </div>
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='brown', icon='coffee', prefix='fa')
            ).add_to(marker_cluster)
        
        # Save map
        map_file = os.path.join(self.viz_dir, f"{chain_name.replace(' ', '_').lower()}_map.html")
        m.save(map_file)
        
        return m
    
    def create_portland_map(self, pdx_df):
        """Create a detailed map of coffee shops in Portland, OR."""
        # Create base map centered on Portland
        m = folium.Map(
            location=[45.5236, -122.6750],
            zoom_start=13,
            tiles='cartodbpositron'
        )
        
        # Create marker clusters for better visualization
        marker_cluster = MarkerCluster().add_to(m)
        
        # Filter for valid coordinates
        valid_coords = pdx_df.dropna(subset=['latitude', 'longitude'])
        
        # Add markers for each coffee shop
        for _, row in valid_coords.iterrows():
            status = "Closed" if pd.notna(row.get('date_closed')) else "Open"
            color = 'red' if status == 'Closed' else 'green'
            
            popup_content = f"""
            <div style="width:200px">
                <b>{row['name']}</b><br>
                Status: {status}<br>
                {row.get('address', 'No address')}<br>
                Portland, {row.get('region', 'OR')} {row.get('postcode', '')}
            </div>
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=color, icon='coffee', prefix='fa')
            ).add_to(marker_cluster)
        
        # Add heatmap layer
        heat_data = [[row['latitude'], row['longitude']] for _, row in valid_coords.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        map_file = os.path.join(self.viz_dir, 'portland_coffee_shops_map.html')
        m.save(map_file)
        
        return m
    
    def create_closure_trend_chart(self, monthly_closures):
        """Create a chart showing coffee shop closure trends over time."""
        if monthly_closures is None or len(monthly_closures) == 0:
            print("No closure data available for trend chart")
            return None
            
        # Create a plotly figure for closures over time
        fig = px.line(
            monthly_closures, 
            x='year_month', 
            y='chain_closures', 
            color='region',
            title='Coffee Shop Closures Over Time by State',
            labels={
                'year_month': 'Date',
                'chain_closures': 'Number of Closures',
                'region': 'State'
            }
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Closures",
            legend_title="State",
            hovermode="x unified",
            plot_bgcolor='white'
        )
        
        # Add hoverable information
        fig.update_traces(
            hovertemplate='%{y} closures<extra></extra>'
        )
        
        # Save the figure
        fig.write_html(os.path.join(self.viz_dir, 'closure_trends.html'))
        fig.write_image(os.path.join(self.viz_dir, 'closure_trends.png'))
        
        return fig
    
    def create_chain_presence_heatmap(self, df, top_chains):
        """Create a heatmap showing chain presence across states."""
        # Get top chain names
        top_chain_names = list(top_chains.index)
        
        # Create a pivot table of chains by state
        state_pivot = pd.crosstab(
            df['region'], 
            df['name']
        )[top_chain_names]
        
        # Select top 15 states by total coffee shops
        top_states = df['region'].value_counts().head(15).index
        state_pivot = state_pivot.loc[top_states]
        
        # Create heatmap using plotly
        fig = px.imshow(
            state_pivot,
            labels=dict(x="Coffee Chain", y="State", color="Location Count"),
            x=top_chain_names,
            y=list(top_states),
            color_continuous_scale='YlOrBr',
            title='Coffee Chain Presence by State'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='white'
        )
        
        # Save the figure
        fig.write_html(os.path.join(self.viz_dir, 'chain_state_heatmap.html'))
        fig.write_image(os.path.join(self.viz_dir, 'chain_state_heatmap.png'))
        
        return fig
    
    def create_dashboard(self, results):
        """Create a comprehensive dashboard with multiple visualizations."""
        # Extract data from results
        top_chains = results.get('top_chains')
        status_df = results.get('status_df')
        monthly_closures = results.get('monthly_closures')
        
        if not all([top_chains is not None, status_df is not None]):
            print("Insufficient data for dashboard creation")
            return None
            
        # Create a plotly dashboard with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top Coffee Chains by Location Count',
                'Open vs Closed Status',
                'Coffee Shop Closures Over Time',
                'Chain Distribution by State'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "choropleth", "colspan": 1}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # 1. Top chains bar chart
        chains_df = pd.DataFrame({
            'Chain': top_chains.index,
            'Count': top_chains.values
        })
        
        fig.add_trace(
            go.Bar(
                x=chains_df['Chain'],
                y=chains_df['Count'],
                marker_color='#8B4513',
                name='Locations'
            ),
            row=1, col=1
        )
        
        # 2. Open vs Closed status (stacked bar)
        for status, color in zip(['Open', 'Closed'], ['#2E8B57', '#CD5C5C']):
            status_data = status_df[status_df['status'] == status]
            fig.add_trace(
                go.Bar(
                    x=status_data['name'],
                    y=status_data['count'],
                    name=status,
                    marker_color=color
                ),
                row=1, col=2
            )
        
# 3. Closures over time (if available)
            if monthly_closures is not None and len(monthly_closures) > 0:
                for state in monthly_closures['region'].unique():
                    state_data = monthly_closures[monthly_closures['region'] == state]
                    fig.add_trace(
                        go.Scatter(
                            x=state_data['year_month'],
                            y=state_data['chain_closures'],
                            mode='lines+markers',
                            name=state,
                            hovertemplate='%{y} closures<extra>%{x}</extra>'
                        ),
                        row=2, col=1
                    )
            else:
                fig.add_annotation(
                    text="No closure data available",
                    xref="x3", yref="y3",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14)
                )
            
            # 4. State-level data (choropleth)
            state_counts = self.df_pd['region'].value_counts().reset_index()
            state_counts.columns = ['state', 'count']
            
            fig.add_trace(
                go.Choropleth(
                    locations=state_counts['state'],
                    z=state_counts['count'],
                    locationmode='USA-states',
                    colorscale='YlOrBr',
                    marker_line_color='white',
                    marker_line_width=0.5,
                    colorbar_title="Count",
                    name='Coffee Shop Count'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="Coffee Shops Dashboard",
                height=900,
                width=1200,
                barmode='stack',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5
                ),
                font=dict(family="Arial, sans-serif", size=10)
            )
            
            # Update x-axis
            fig.update_xaxes(title_text="Coffee Chain", row=1, col=1, tickangle=-45)
            fig.update_xaxes(title_text="Coffee Chain", row=1, col=2, tickangle=-45)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            # Update y-axis
            fig.update_yaxes(title_text="Number of Locations", row=1, col=1)
            fig.update_yaxes(title_text="Number of Locations", row=1, col=2)
            fig.update_yaxes(title_text="Number of Closures", row=2, col=1)
            
            # Update geo layout
            fig.update_geos(
                scope='usa',
                projection_type='albers usa',
                showlakes=True,
                lakecolor='rgb(255, 255, 255)',
                row=2, col=2
            )
            
            # Save the dashboard
            fig.write_html(os.path.join(self.viz_dir, 'coffee_dashboard.html'))
            fig.write_image(os.path.join(self.viz_dir, 'coffee_dashboard.png'))
            
            return fig
    
    def create_state_comparison_chart(self, state_data, metric="count"):
        """Create a chart comparing coffee shops across states."""
        if state_data is None or len(state_data) == 0:
            print("No state data available for comparison chart")
            return None
        
        # Sort data by metric
        state_data = state_data.sort_values(metric, ascending=False)
        
        # Create chart
        fig = px.bar(
            state_data,
            x='state',
            y=metric,
            title=f'Coffee Shops by State ({metric.title()})',
            color=metric,
            color_continuous_scale='YlOrBr',
            labels={
                'state': 'State',
                metric: metric.title()
            }
        )
        
        fig.update_layout(
            xaxis_title="State",
            yaxis_title=metric.title(),
            xaxis_tickangle=-45,
            plot_bgcolor='white'
        )
        
        # Save the figure
        fig.write_html(os.path.join(self.viz_dir, f'state_comparison_{metric}.html'))
        fig.write_image(os.path.join(self.viz_dir, f'state_comparison_{metric}.png'))
        
        return fig
    
    def create_chain_comparison_scatter(self, df):
        """Create a scatter plot comparing coffee chains by various metrics."""
        # Group by chain and calculate metrics
        chain_metrics = df.groupby('name').agg(
            locations=('name', 'size'),
            states=('region', 'nunique'),
            avg_lat=('latitude', 'mean'),
            avg_lon=('longitude', 'mean')
        ).reset_index()
        
        # Calculate regional coverage
        chain_metrics['regional_coverage'] = chain_metrics['states'] / 50 * 100
        
        # Calculate closure rate if data is available
        if 'date_closed' in df.columns:
            closure_counts = df[df['date_closed'].notna()].groupby('name').size()
            for chain in chain_metrics['name']:
                total = chain_metrics.loc[chain_metrics['name'] == chain, 'locations'].values[0]
                closed = closure_counts.get(chain, 0)
                chain_metrics.loc[chain_metrics['name'] == chain, 'closure_rate'] = (closed / total * 100) if total > 0 else 0
        
        # Create scatter plot
        fig = px.scatter(
            chain_metrics.sort_values('locations', ascending=False).head(20),
            x='locations',
            y='regional_coverage',
            size='locations',
            color='regional_coverage',
            hover_name='name',
            color_continuous_scale='YlOrBr',
            title='Coffee Chains: Total Locations vs. Regional Coverage',
            labels={
                'locations': 'Number of Locations',
                'regional_coverage': 'US State Coverage (%)'
            }
        )
        
        fig.update_layout(
            xaxis_title="Number of Locations",
            yaxis_title="State Coverage (%)",
            plot_bgcolor='white'
        )
        
        # Save the figure
        fig.write_html(os.path.join(self.viz_dir, 'chain_comparison_scatter.html'))
        fig.write_image(os.path.join(self.viz_dir, 'chain_comparison_scatter.png'))
        
        return fig
    
    def create_regional_maps(self, df, regions=None):
        """Create detailed maps for specific regions."""
        if regions is None:
            # Default to top 5 states by coffee shop count
            regions = df['region'].value_counts().head(5).index
        
        maps = {}
        for region in regions:
            print(f"Creating map for {region}...")
            
            # Filter data for the region
            region_df = df[df['region'] == region].copy()
            
            # Skip if no valid data
            if len(region_df) == 0 or region_df['latitude'].isna().all():
                print(f"No valid data for {region}")
                continue
            
            # Calculate center of locations
            center_lat = region_df['latitude'].mean()
            center_lon = region_df['longitude'].mean()
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=7,
                tiles='cartodbpositron'
            )
            
            # Create marker cluster
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add markers for all locations
            for _, row in region_df.dropna(subset=['latitude', 'longitude']).iterrows():
                status = "Closed" if pd.notna(row.get('date_closed')) else "Open"
                color = 'red' if status == 'Closed' else 'green'
                
                popup_content = f"""
                <div style="width:200px">
                    <b>{row['name']}</b><br>
                    Status: {status}<br>
                    {row.get('address', 'No address')}<br>
                    {row.get('locality', '')}, {row.get('region', '')} {row.get('postcode', '')}
                </div>
                """
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color=color, icon='coffee', prefix='fa')
                ).add_to(marker_cluster)
            
            # Add heatmap layer
            heat_data = [
                [row['latitude'], row['longitude']] 
                for _, row in region_df.dropna(subset=['latitude', 'longitude']).iterrows()
            ]
            
            HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map
            map_file = os.path.join(self.viz_dir, f'{region.lower()}_coffee_shops_map.html')
            m.save(map_file)
            
            maps[region] = m
            print(f"Saved {region} map with {len(region_df)} locations")
        
        return maps
    
    def create_summary_visualization(self, results):
        """Create a comprehensive visualization package from all results."""
        print("\nCreating comprehensive visualization package...")
        
        # Get data from results
        top_chains = results.get('top_chains')
        status_df = results.get('status_df')
        monthly_closures = results.get('monthly_closures')
        
        # Create all visualizations
        visualizations = {}
        
        # 1. Chain bar chart
        if top_chains is not None:
            visualizations['chain_bar'] = self.create_chain_bar_chart(top_chains)
        
        # 2. Open vs closed chart
        if status_df is not None:
            visualizations['status_chart'] = self.create_open_closed_chart(status_df)
        
        # 3. State choropleth
        if 'df' in results:
            visualizations['state_map'] = self.create_state_choropleth(results['df'])
        
        # 4. Closure trend chart
        if monthly_closures is not None:
            visualizations['closure_trend'] = self.create_closure_trend_chart(monthly_closures)
        
        # 5. Combined dashboard
        visualizations['dashboard'] = self.create_dashboard(results)
        
        print(f"Created {len(visualizations)} visualizations in {self.viz_dir}")
        
        return visualizations


if __name__ == "__main__":
    # Quick demo if this module is run directly
    import pandas as pd
    
    print("Coffee Shop Visualization Module")
    print("This module is intended to be imported by the main analysis script.")
    
    # Create sample data for testing
    sample_data = {
        'top_chains': pd.Series({
            'Dutch Bros Coffee': 345,
            'Peet\'s Coffee': 290,
            'Dunkin Donuts Coffee Shop': 245,
            'La Colombe Coffee': 180,
            'Blue Bottle Coffee': 110
        }),
        'status_df': pd.DataFrame({
            'name': ['Dutch Bros Coffee', 'Dutch Bros Coffee', 'Peet\'s Coffee', 'Peet\'s Coffee'],
            'status': ['Open', 'Closed', 'Open', 'Closed'],
            'count': [300, 45, 250, 40]
        })
    }
    
    # Create visualizer and generate sample visualizations
    viz = CoffeeShopVisualizer(output_dir="./sample_output")
    viz.create_chain_bar_chart(sample_data['top_chains'])
    viz.create_open_closed_chart(sample_data['status_df'])
    
    print("Sample visualizations created in ./sample_output/visualizations/")