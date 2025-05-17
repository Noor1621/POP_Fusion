import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from io import BytesIO
import base64

# Initialize Earth Engine
ee.Initialize()

# -------------------------------
# Population Dataset Setup
# -------------------------------
datasets = {
    "CIESIN/GPWv411/GPW_UNWPP-Adjusted_Population_Density": {
        "label": "GPW v4.11 Population Density",
        "band": "unwpp-adjusted_population_density",
        "years": [2000, 2005, 2010, 2015, 2020]
    },
    "LandScan_Global": {
        "label": "LandScan Global",
        "band": "b1",
        "years": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
        "dynamic_id": "projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL/landscan-global-{year}"
    },
    "WorldPop/GP/100m/pop": {
        "label": "WorldPop 100m",
        "band": "population",
        "years": [2010, 2015, 2020]
    },
    "JRC/GHSL/P2016/POP_GPW_GLOBE_V1": {
        "label": "GHS-POP",
        "band": "population",
        "years": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    },
    "JRC/GHSL/P2023A/GHS_POP": {
        "label": "GHSL Global Population Surfaces (P2023A)",
        "band": "population_count",
        "years": list(range(1975, 2031))
    },
}

bin_thresholds = [1, 6, 26, 51, 101, 501, 2501, 5001, 185000]
legend_colors = ['#ffffcc', '#fff775', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026', '#800026']
legend_labels = ['1-5', '6-25', '26-50', '51-100', '101-500', '501-2500', '2501-5000', '5001-185000']

vis_params = {
    'min': 1,
    'max': len(legend_colors),
    'palette': legend_colors
}

def classify_image(image):
    bins = bin_thresholds
    classified = image.gt(bins[0]).And(image.lte(bins[1])).multiply(1)
    for i in range(1, len(bins) - 1):
        classified = classified.where(image.gt(bins[i]).And(image.lte(bins[i + 1])), i + 1)
    return classified

def load_dataset_by_year(dataset_id, band, year):
    if dataset_id == "LandScan_Global":
        image_id = datasets[dataset_id]["dynamic_id"].format(year=year)
        return ee.Image(image_id).select(band)
    else:
        collection = ee.ImageCollection(dataset_id).filterDate(f'{year}-01-01', f'{year+1}-01-01')
        return collection.select(band).first()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("PopFusion")

dataset_name = st.selectbox(
    "Select Dataset", 
    [(v["label"], k) for k, v in datasets.items()], 
    format_func=lambda x: x[0]
)
dataset_id = dataset_name[1]
selected_dataset = datasets[dataset_id]

year = st.selectbox("Select Year", selected_dataset["years"])

Map = geemap.Map()
Map.add_basemap("HYBRID")
Map.set_center(0, 20, 2)

geometry = ee.Geometry.Polygon(
    [[[-10.0, 5.0], [10.0, 5.0], [10.0, 20.0], [-10.0, 20.0], [-10.0, 5.0]]]
)

image = load_dataset_by_year(dataset_id, selected_dataset["band"], year)
if image:
    image = image.clip(geometry).updateMask(image.gt(0))
    classified = classify_image(image)
    Map.addLayer(classified, vis_params, f"{selected_dataset['label']} ({year})")

    # Add legend
    legend_dict = {label: color for label, color in zip(legend_labels, legend_colors)}
    Map.add_legend(title="Population Count", legend_dict=legend_dict)

    with st.expander("🗺️ Interactive Map"):
        Map.to_streamlit(height=500)

    stats = image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=1000,
        maxPixels=1e12
    ).getInfo()

    pop_total = stats.get(selected_dataset["band"])
    if pop_total:
        st.success(f"✅ Total Population in Selected Area ({year}): {pop_total:,.0f}")
    else:
        st.warning("⚠️ No data found for the selected region.")
st.markdown("---")
col1, col2 = st.columns([0.8, 0.2])

with col1:
    st.markdown("© 2025 **PopFusion**. All rights reserved.  \nPowered by [Google Earth Engine/API](https://earthengine.google.com/) & [Streamlit](https://streamlit.io/)")








# -------------------------------

