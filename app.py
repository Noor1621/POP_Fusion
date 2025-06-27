import streamlit as st
import ee
import geemap.foliumap as geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import io
import time
from shapely.geometry import Polygon, MultiPolygon
import plotly.express as px
import plotly.graph_objects as go

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    st.error("Earth Engine initialization failed. Please authenticate first.")
    st.stop()

datasets = {
    "CIESIN/GPWv411/GPW_UNWPP-Adjusted_Population_Count": {
        "label": "GPW v4.11 Population Count",
        "band": "unwpp-adjusted_population_count",
        "years": [2000, 2005, 2010, 2015, 2020]
    },
    "LandScan_Global": {
        "label": "LandScan Global",
        "band": "b1",
        "years": [2000, 2005, 2010, 2015, 2020, 2021],
        "dynamic_id": "projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL/landscan-global-{year}"
    },
    "WorldPop/GP/100m/pop": {
        "label": "WorldPop 100m",
        "band": "population",
        "years": [2010, 2015, 2020]
    }
}

legend_colors = ['#ffffcc', '#fff775', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026', '#800026']
bin_thresholds = [1, 6, 26, 51, 101, 501, 2501, 5001, 185000]

def classify_image(image):
    """Classify population image into bins"""
    bins = bin_thresholds
    classified = image.gt(bins[0]).And(image.lte(bins[1])).multiply(1)
    for i in range(1, len(bins) - 1):
        classified = classified.where(image.gt(bins[i]).And(image.lte(bins[i + 1])), i + 1)
    return classified

def load_dataset_by_year(dataset_id, band, year):
    """Load dataset for specific year with better error handling"""
    try:
        if dataset_id == "LandScan_Global":
            image_id = datasets[dataset_id]["dynamic_id"].format(year=year)
            image = ee.Image(image_id).select(band)
        else:
            # Use more flexible date filtering
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            collection = ee.ImageCollection(dataset_id).filterDate(start_date, end_date)
            
            # Check if collection has images
            size = collection.size().getInfo()
            if size == 0:
                # Try to get closest year
                collection = ee.ImageCollection(dataset_id)
                image = collection.select(band).first()
            else:
                image = collection.select(band).first()
        
        return image
    except Exception as e:
        st.warning(f"Could not load {dataset_id} for year {year}: {str(e)}")
        return None

def get_population_for_geometry(dataset_id, band, year, geometry):
    """Get population statistics for a specific geometry and dataset"""
    try:
        image = load_dataset_by_year(dataset_id, band, year)
        if image is None:
            return None
        
        clipped = image.clip(geometry)
        stats = clipped.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=1000,
            maxPixels=1e12,
            bestEffort=True
        ).getInfo()
        
        return stats.get(band, None)
    except Exception as e:
        st.warning(f"Error getting population for {dataset_id}: {str(e)}")
        return None

def plot_population_comparison(pop_data_df, selected_year):
    """Create interactive bar chart comparison"""
    fig = go.Figure()
    
    colors = ['#4c72b0', '#55a868', '#c44e52']
    
    for i, (dataset, pop) in enumerate(zip(pop_data_df['Dataset'], pop_data_df['Population'])):
        fig.add_bar(
            x=[dataset],
            y=[pop],
            name=dataset,
            marker_color=colors[i % len(colors)],
            text=[f'{int(pop):,}'],
            textposition='outside'
        )
    
    fig.update_layout(
        title=f"Total Population Comparison by Dataset ({selected_year})",
        xaxis_title="Population Dataset",
        yaxis_title="Population Count",
        showlegend=False,
        height=500
    )
    
    return fig

def create_population_animation(geometry, area_name):
    """Create animated chart showing population trends across years"""
    # Collect all available years from all datasets
    all_years = set()
    for dataset_info in datasets.values():
        all_years.update(dataset_info["years"])
    all_years = sorted(list(all_years))
    
    # Collect population data for all datasets and years
    animation_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(datasets) * len(all_years)
    current_step = 0
    
    for dataset_id, dataset_info in datasets.items():
        for year in all_years:
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            status_text.text(f"Loading {dataset_info['label']} for {year}...")
            
            if year in dataset_info["years"]:
                pop = get_population_for_geometry(dataset_id, dataset_info["band"], year, geometry)
                if pop is not None and pop > 0:
                    animation_data.append({
                        'Year': year,
                        'Dataset': dataset_info['label'],
                        'Population': pop
                    })
    
    progress_bar.empty()
    status_text.empty()
    
    if not animation_data:
        st.warning("No population data available for animation")
        return None
    
    df = pd.DataFrame(animation_data)
    
    # Create animated line chart
    fig = px.line(
        df, 
        x='Year', 
        y='Population', 
        color='Dataset',
        title=f'Population Trends Over Time - {area_name}',
        markers=True,
        height=600
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Population Count",
        hovermode='x unified'
    )
    
    return fig, df

def main():
    st.set_page_config(page_title="POP FUSION", layout="wide")
    st.title("🌍 POP FUSION")
    
    # Sidebar controls
    st.sidebar.header("Dataset Selection")
    dataset_key = st.sidebar.selectbox(
        "Select Dataset", 
        options=list(datasets.keys()), 
        format_func=lambda x: datasets[x]["label"]
    )
    
    years = datasets[dataset_key]["years"]
    year = st.sidebar.selectbox("Select Year", options=years)
    
    # File upload
    st.sidebar.header("Area Selection")
    geojson_file = st.file_uploader(
        "Upload GeoJSON file defining area(s) of interest (polygon)", 
        type=["geojson", "json"]
    )
    
    geometry = None
    gdf = None
    area_name = ""
    
    if geojson_file:
        try:
            gdf = gpd.read_file(geojson_file)
            
            # Display columns
            with st.expander("GeoJSON File Info"):
                st.write("Columns detected in the uploaded GeoJSON:")
                st.write(list(gdf.columns))
                st.write(f"Number of areas: {len(gdf)}")
            
            # Auto-detect name column
            default_name_cols = [c for c in gdf.columns if c.lower() in ['name', 'country', 'admin', 'admin0name', 'sovereignt']]
            default_col = default_name_cols[0] if default_name_cols else None
            
            name_field = st.selectbox(
                "Select column to use as area name:",
                options=[None] + list(gdf.columns),
                index=(list(gdf.columns).index(default_col) + 1 if default_col else 0)
            )
            
            if name_field is None:
                gdf['area_name'] = [f'Area {i+1}' for i in range(len(gdf))]
                name_field = 'area_name'
            
            selected_area = st.selectbox(
                "Select an area from uploaded file:",
                options=gdf.index,
                format_func=lambda i: str(gdf.loc[i, name_field])
            )
            
            area_name = str(gdf.loc[selected_area, name_field])
            selected_geom = gdf.geometry[selected_area]
            
            # Convert geometry to Earth Engine format
            if isinstance(selected_geom, Polygon):
                coords = list(selected_geom.exterior.coords)
                geometry = ee.Geometry.Polygon(coords)
            elif isinstance(selected_geom, MultiPolygon):
                multi_coords = []
                for poly in selected_geom.geoms:
                    multi_coords.append(list(poly.exterior.coords))
                geometry = ee.Geometry.MultiPolygon(multi_coords)
            else:
                st.error("Uploaded geometry is not Polygon or MultiPolygon.")
                st.stop()
            
            # Show map
            gdf['centroid'] = gdf.geometry.centroid
            gdf['latitude'] = gdf['centroid'].y
            gdf['longitude'] = gdf['centroid'].x
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"Selected Area: {area_name}")
            with col2:
                st.map(gdf[['latitude', 'longitude']])
                
        except Exception as e:
            st.error(f"Error reading GeoJSON file: {str(e)}")
            st.stop()
    
    if geometry:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Current Year Analysis", "📈 Population Trends", "🗺️ Map Visualization"])
        
        with tab1:
            st.subheader(f"Population Analysis for {area_name} ({year})")
            
            # Get population data for all datasets for the selected year
            pop_data = []
            
            with st.spinner("Loading population data..."):
                for ds_key, ds_info in datasets.items():
                    if year in ds_info["years"]:
                        pop = get_population_for_geometry(ds_key, ds_info["band"], year, geometry)
                        if pop is not None and pop > 0:
                            pop_data.append({
                                'Dataset': ds_info['label'],
                                'Population': pop
                            })
            
            if pop_data:
                df_comparison = pd.DataFrame(pop_data)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                for i, row in df_comparison.iterrows():
                    with [col1, col2, col3][i]:
                        st.metric(
                            label=row['Dataset'],
                            value=f"{int(row['Population']):,}",
                            delta=None
                        )
                
                # Interactive bar chart
                fig_bar = plot_population_comparison(df_comparison, year)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Data table
                st.subheader("Population Data Summary")
                st.dataframe(df_comparison, use_container_width=True)
                
            else:
                st.warning(f"No population data available for {year}")
        
        with tab2:
            st.subheader(f"Population Trends Over Time - {area_name}")
            
            if st.button("Generate Population Animation", type="primary"):
                with st.spinner("Creating population trends visualization..."):
                    fig_animation, trend_data = create_population_animation(geometry, area_name)
                    
                    if fig_animation:
                        st.plotly_chart(fig_animation, use_container_width=True)
                        
                        # Show trend data
                        st.subheader("Trend Data")
                        pivot_data = trend_data.pivot(index='Year', columns='Dataset', values='Population')
                        st.dataframe(pivot_data, use_container_width=True)
                        
                        # Download option
                        csv = trend_data.to_csv(index=False)
                        st.download_button(
                            label="Download Trend Data as CSV",
                            data=csv,
                            file_name=f"population_trends_{area_name.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
        
        with tab3:
            st.subheader(f"Population Distribution Map - {area_name} ({year})")
            
            with st.spinner("Loading map visualization..."):
                image = load_dataset_by_year(dataset_key, datasets[dataset_key]["band"], year)
                
                if image is not None:
                    clipped = image.clip(geometry)
                    classified = classify_image(clipped)
                    
                    # Create map
                    center_lat = gdf.loc[selected_area].geometry.centroid.y
                    center_lon = gdf.loc[selected_area].geometry.centroid.x
                    
                    Map = geemap.Map(center=[center_lat, center_lon], zoom=8)
                    Map.addLayer(
                        classified, 
                        {'min': 1, 'max': len(legend_colors), 'palette': legend_colors},
                        f"{datasets[dataset_key]['label']} ({year})"
                    )
                    Map.add_legend(
                        title="Population Count",
                        labels=['1-5', '6-25', '26-50', '51-100', '101-500', '501-2500', '2501-5000', '5001+'],
                        colors=legend_colors
                    )
                    
                    Map.to_streamlit(height=600)
                    
                    # Show total population for selected dataset
                    pop_total = get_population_for_geometry(dataset_key, datasets[dataset_key]["band"], year, geometry)
                    if pop_total:
                        st.success(f"**Total population in {area_name}: {int(pop_total):,}**")
                else:
                    st.error("Could not load map visualization for the selected dataset and year.")
    
    else:
        st.info("👆 Please upload a GeoJSON polygon file to define the area of interest and start the analysis.")
        
        # Show example of what the app can do
        st.subheader("What this app can do:")
        st.markdown("""
        - 📊 **Compare population datasets** for any geographic area
        - 📈 **Visualize population trends** across different years  

        - 📱 **Export data** for further analysis
        
        **Supported datasets:**
        - GPW v4.11 Population Count (2000-2020)
        - LandScan Global (2014-2021) 
        - WorldPop 100m (2010, 2015, 2020)
        """)

if __name__ == "__main__":
    main()