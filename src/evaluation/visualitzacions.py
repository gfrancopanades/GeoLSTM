#!/usr/bin/env python
# coding: utf-8

# ### Libraries import
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import warnings

import folium
import branca.colormap as cm

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import glob

from src.data_file_names import *
from src.evaluation.utils import *
from src.data_file_names import *
from src.training.utils import *
from src.training.io import *
from src.data.data_utils import *

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Specify the folder name relative to the working directory
iter_num=14
folder_name = f"vels_ints_weig_iter{iter_num}"  # Replace with your folder name
folder_name="combi_3"
folder_results = os.path.join(os.path.dirname(os.path.join(os.getcwd())), 'results', folder_name)

# List to hold individual DataFrames
dataframes = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_results):
    print(file_name)
    if file_name.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(folder_results, file_name)
        # Read the CSV file with specified separator, decimal, and encoding
        df = pd.read_csv(file_path, sep=";", decimal=".", encoding="latin1")
        # Add a new column with the file name
        df['source_file'] = file_name
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames into one
final_df = pd.concat(dataframes, ignore_index=True)

final_df['Model'] = final_df['source_file'].str.replace("predictions_model_","")
final_df['Model'] = final_df['Model'].str.replace("_granularity_1h_projection_1hour_","_")

for iter_num in [13, 14, 15, 16, 17]:
    dict_arxiu_nom = {
    f'ConvTFTModel_iter{iter_num}_YES_geo_YES_vw.csv': f'GeoConv-TFT_{iter_num}',
    f'ConvTransformerModel_iter{iter_num}_YES_geo_YES_vw.csv': f'GeoConv-Transformer_{iter_num}',
    f'Conv_LSTM_Model_iter{iter_num}_YES_geo_YES_vw.csv': f'GeoConv-LSTM_{iter_num}',
    f'DoubleConvLSTM_iter{iter_num}_YES_geo_YES_vw.csv': f'Geo2Conv-LSTM_{iter_num}',
    f'DoubleConvTFTModel_iter{iter_num}_YES_geo_YES_vw.csv': f'Geo2Conv-TFT_{iter_num}',
    f'DoubleConvTransformerModel_iter{iter_num}_YES_geo_YES_vw.csv': f'Geo2Conv-Transformer_{iter_num}',
    f'LSTMAttModel_iter{iter_num}_NO_geo_YES_vw.csv': f'Attention-LSTM_{iter_num}',
    f'LSTMAttModel_iter{iter_num}_YES_geo_YES_vw.csv': f'GeoAttention-LSTM_{iter_num}',
    f'LSTMModel_iter{iter_num}_NO_geo_YES_vw.csv': f'LSTM_{iter_num}',
    f'LSTMModel_iter{iter_num}_YES_geo_YES_vw.csv': f'GeoLSTM_{iter_num}',
    f'SARIMA_seasonality=daily_iter{iter_num}.csv': f'SARIMA_{iter_num}',
    f'TFTModel_iter{iter_num}_NO_geo_YES_vw.csv': f'TFT_{iter_num}',
    f'TFTModel_iter{iter_num}_YES_geo_YES_vw.csv': f'GeoTFT_{iter_num}',
    f'TransformerModel_iter{iter_num}_NO_geo_YES_vw.csv': f'Transformer_{iter_num}',
    f'TransformerModel_iter{iter_num}_YES_geo_YES_vw.csv': f'GeoTransformer_{iter_num}',
    f'xLSTMModel_iter{iter_num}_NO_geo_YES_vw.csv': f'X-LSTM_{iter_num}',
    f'xLSTMModel_iter{iter_num}_YES_geo_YES_vw.csv': f'GeoX-LSTM_{iter_num}'
    }

    final_df['Model'] = final_df['Model'].replace(dict_arxiu_nom)
    print(f"Model unique values: {final_df['Model'].unique()}")

final_df['dat'] = pd.to_datetime(final_df['dat'], errors='coerce')
final_df['Fine-tune Time'] = pd.to_datetime(final_df['Fine-tune Time'], errors='coerce')
final_df['horizon_prediction']=final_df['dat']-final_df['Fine-tune Time']
final_df['via']='AP-7'
final_df['sen']='dec'
final_df['Model'].unique()

# Define the columns to process
columns_to_process = ['mean_speed', 'std_dev_speed', 'percentile_10', 'percentile_85', 'intTot', 'intP']

final_df['mean_speed'] = final_df['mean_speed'].round(decimals=0)
final_df['std_dev_speed'] = final_df['std_dev_speed'].round(decimals=4)
final_df['percentile_10'] = final_df['percentile_10'].round(decimals=0)
final_df['percentile_85'] = final_df['percentile_85'].round(decimals=0)
final_df['intTot'] = final_df['intTot'].round(decimals=0)
final_df['intP'] = final_df['intP'].round(decimals=0)

# Perform the subtraction and store the result in the original column names
for col in columns_to_process:
    final_df[f"{col}_residuals"] = final_df[f"{col}_pred"] - final_df[col]
    final_df[f"{col}_residuals_abs"] = final_df[f"{col}_residuals"].abs()

final_df['mean_speed_ep']=final_df['mean_speed_residuals']/final_df['mean_speed']
final_df['intTot_ep']=final_df['intTot_residuals']/final_df['intTot']
final_df['intP_ep']=final_df['intP_residuals']/final_df['intP']


# ### Data Treatment Dupls
unique_cols = ['Model', 'Any', 'mes', 'dia', 'diaSem', 'hor', 'via', 'pk', 'sen']
final_df = final_df.sort_values('mean_speed_residuals_abs', ascending=True)
final_df = final_df.drop_duplicates(subset=unique_cols, keep='first')
final_df = final_df[final_df.dat<=datetime.datetime(2023, 12, 30, 23, 0, 0)]
final_df = final_df[final_df.dat>=datetime.datetime(2023, 9, 3, 0, 0, 0)]

df_og = pd.read_csv(os.path.join(folder_dades, data_version + '_vel_int_cal_mob_1h_new_etds_5.csv'), sep=";", decimal=".", encoding='latin-1')

print(df_og[['mean_speed', 'std_dev_speed', 'percentile_10', 'percentile_85','intTot', 'intP']].describe())

# ### Error Tables Comparison
print("mean_speed:")
print_error_table_from_dataframe(final_df.rename(columns={'mean_speed':'Actual','mean_speed_pred':'Predicted','mean_speed_residuals':'Residuals','mean_speed_residuals_abs':'Absolute Residuals'}), actual_column='Actual', predicted_column='Predicted', model_column='Model',variab='mean_speed')

print("====================================================")
print("percentile_10:")
print_error_table_from_dataframe(final_df.rename(columns={'percentile_10':'Actual','percentile_10_pred':'Predicted','percentile_10_residuals':'Residuals','percentile_10_residuals_abs':'Absolute Residuals'}), actual_column='Actual', predicted_column='Predicted', model_column='Model',variab='percentile_10')

print("====================================================")
print("percentile_85:")
print_error_table_from_dataframe(final_df.rename(columns={'percentile_85':'Actual','percentile_85_pred':'Predicted','percentile_85_residuals':'Residuals','percentile_85_residuals_abs':'Absolute Residuals'}), actual_column='Actual', predicted_column='Predicted', model_column='Model',variab='percentile_85')

print("====================================================")
print("std_dev_speed:")
print_error_table_from_dataframe(final_df.rename(columns={'std_dev_speed':'Actual','std_dev_speed_pred':'Predicted','std_dev_speed_residuals':'Residuals','std_dev_speed_residuals_abs':'Absolute Residuals'}), actual_column='Actual', predicted_column='Predicted', model_column='Model',variab='std_dev_speed')

print("====================================================")
print("intTot:")
print_error_table_from_dataframe(final_df.rename(columns={'intTot':'Actual','intTot_pred':'Predicted','intTot_residuals':'Residuals','intTot_residuals_abs':'Absolute Residuals'}), actual_column='Actual', predicted_column='Predicted', model_column='Model', variab='intTot')

print("====================================================")
print("intP:")
print_error_table_from_dataframe(final_df.rename(columns={'intP':'Actual','intP_pred':'Predicted','intP_residuals':'Residuals','intP_residuals_abs':'Absolute Residuals'}), actual_column='Actual', predicted_column='Predicted', model_column='Model',variab='intP')

# New function that RETURNS a DataFrame
# Empty list to collect all results
all_results = []

# List of variables you want to process
variables = ['mean_speed', 'percentile_10', 'percentile_85', 'std_dev_speed', 'intTot', 'intP']

# Loop through each variable
for var in variables:
    temp_df = final_df.rename(columns={
        f'{var}': 'Actual',
        f'{var}_pred': 'Predicted',
        f'{var}_residuals': 'Residuals',
        f'{var}_residuals_abs': 'Absolute Residuals'
    })
    # Get the error table for the current variable
    error_table = get_error_table_from_dataframe(
        temp_df,
        actual_column='Actual',
        predicted_column='Predicted',
        model_column='Model',
        variab=var
    )
    # Add a new column to indicate which variable it is
    error_table['Variable'] = var
    # Collect the result
    all_results.append(error_table)

# Concatenate all results into a single DataFrame
final_error_table = pd.concat(all_results, ignore_index=True)

# Optional: reorder columns
final_error_table = final_error_table[['Variable', 'Model', 'MAE', 'MSE', 'RMSE', 'R2']]

# Save to CSV
final_error_table.to_csv('final_error_table.csv', index=False)

print("All results saved to final_error_table.csv")

# Only selecting the columns you want (no percentiles)
actual_predicted_columns = [
    ('mean_speed', 'mean_speed_pred', 160),
    ('percentile_10', 'percentile_10_pred', 160),
    ('intTot', 'intTot_pred', 15000),
    ('intP', 'intP_pred', 3000)
]

# Corresponding titles
variables = [
    'mean_speed_residuals_abs', 
    'percentile_10_residuals_abs', 
    'intTot_residuals_abs', 
    'intP_residuals_abs'
]

# Títulos para cada gráfico
heatmap_titles = [
    'Mean Speed',
    '10th Percentile Speed',
    'Total Intensity',
    'HWV Intensity']

# Define custom colormap: from pale blue to dark blue
colors = ["#ADD8E6", "#00008B"]  # LightBlue -> DarkBlue
colors = ["#d8ecf3", "#00008B"]  # LightBlue -> DarkBlue
# colors = ["#e6ffff", "#00008B"]  # LightBlue -> DarkBlue

custom_blue_cmap = LinearSegmentedColormap.from_list("pale_to_dark_blue", colors)


models_to_plot = ['SARIMA_13']
filtered_df = final_df[final_df['Model'].isin(models_to_plot)]

fig, axes = plt.subplots(1, 4, figsize=(14, 12))
axes = axes.flatten()

for i, ((actual_col, pred_col, lim), title) in enumerate(zip(actual_predicted_columns, heatmap_titles)):
    ax = axes[i]
    model_data = filtered_df.copy()
    
    sns.histplot(
        x=model_data[actual_col],
        y=model_data[pred_col],
        cbar_kws={"shrink": 0.2},
        bins=100, pmax=1, ax=ax, cbar=True, cmap=custom_blue_cmap                       
    )
    
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Observed', fontsize=13)
    ax.set_ylabel('Predicted', fontsize=13)
    ax.set_aspect('equal')


plt.tight_layout()
plt.show()


# ### Observed Values Histogram
# Definir las combinaciones de columnas a graficar (actual, predicted, límite máximo)
actual_predicted_columns = [
    ('mean_speed', 'mean_speed_pred', 160),
    ('percentile_10', 'percentile_10_pred', 160),
    ('percentile_85', 'percentile_85_pred', 160),
    ('intTot', 'intTot_pred', 10000),
    ('intP', 'intP_pred', 2500),
    ('std_dev_speed', 'std_dev_speed_pred', 60)
]

# Títulos para cada gráfico
hist_titles = [
    'Mean Speed',
    '10th Percentile Speed',
    '85th Percentile Speed',
    'Total Intensity',
    'Heavy-weight Intensity',
    'Speed Std Dev'
]

models_to_plot = ['GeoLSTM_13']
filtered_df = final_df[final_df['Model'].isin(models_to_plot)]

# Crear subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Recorrer cada combinación y generar histogramas superpuestos
for i, ((actual_col, pred_col, lim), title) in enumerate(zip(actual_predicted_columns, hist_titles)):
    ax = axes[i]
    model_data = filtered_df.copy()

    # Histogramas superpuestos: actual (azul), predicho (naranja)
    sns.histplot(
        model_data[actual_col], kde=True, bins=100, stat="density",
        label='Actual', color='C0', alpha=0.5, ax=ax
    )
    sns.histplot(
        model_data[pred_col], kde=True, bins=100, stat="density",
        label='Predicted', color='C1', alpha=0.5, ax=ax
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

plt.tight_layout()
plt.show()

model_name = "GeoLSTM_13"

# Load lon-lat data
df_lon_lat = pd.read_csv("lonlat_pks_ap7_120_220.csv", sep=",", decimal=".", encoding="latin1")
df_lon_lat['pk_rounded'] = np.trunc(df_lon_lat['pk'])
df_lon_lat = df_lon_lat.rename(columns={'pk':'pk_decimal'}).rename(columns={'pk_rounded':'pk'})
df_lon_lat = df_lon_lat[['pk', 'pk_decimal', 'lon', 'lat']].copy()


# Group and calculate MAE
mae_df = final_df[final_df['Model'] == model_name].groupby('pk')[[ 
    'mean_speed_residuals_abs',
    'std_dev_speed_residuals_abs',
    'percentile_10_residuals_abs',
    'percentile_85_residuals_abs',
    'intTot_residuals_abs',
    'intP_residuals_abs'
]].agg(mae).reset_index(drop=False)

# Merge with lon-lat
df_lon_lat = df_lon_lat.merge(mae_df, on='pk')
df_lon_lat = df_lon_lat.sort_values(by='pk_decimal')

# Variables to plot
variables = [
    'mean_speed_residuals_abs', 
    'percentile_10_residuals_abs', 
    'intTot_residuals_abs', 
    'intP_residuals_abs'
]

# Prettier names for map captions
pretty_names = {
    'mean_speed_residuals_abs': 'Mean Speed Residuals (Absolute)',
    'percentile_10_residuals_abs': '10th Percentile Speed Residuals (Absolute)',
    'intTot_residuals_abs': 'Total Intensity Residuals (Absolute)',
    'intP_residuals_abs': 'Heavy-Weight Vehicle Intensity Residuals (Absolute)'
}

# Custom legend (colorbar) value ranges
custom_ranges = {
    'mean_speed_residuals_abs': (2, 25),    # Adjust these as you want
    'percentile_10_residuals_abs': (2, 25),
    'intTot_residuals_abs': (450, 740),
    'intP_residuals_abs': (150, 215)
}

# Center the map
map_center = [df_lon_lat['lat'].mean(), df_lon_lat['lon'].mean()]

for var in variables:
    print(f"Generating map for {pretty_names[var]}...")
    
    # Create the map
    m = folium.Map(location=map_center, zoom_start=7, tiles='cartodbpositron')
    
    # Use custom min and max
    min_val, max_val = custom_ranges[var]
    
    # Create traffic light color scale
    traffic_light_colormap = cm.LinearColormap(
        colors=['green', 'yellow', 'red'],
        vmin=min_val, vmax=max_val
    )
    traffic_light_colormap.caption = pretty_names[var]
    # Add a custom style to make the colorbar bigger    

    traffic_light_colormap.add_to(m)

    # Plot each point
    for _, row in df_lon_lat.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color=traffic_light_colormap(row[var]),
            fill=True,
            fill_color=traffic_light_colormap(row[var]),
            fill_opacity=0.7,
            popup=f"{pretty_names[var]}: {row[var]:.2f}"
        ).add_to(m)

    # Save each map
    file_safe_name = pretty_names[var].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    model_safe_name = model_name.lower()
    filename = f"map_{file_safe_name}_{model_safe_name}.html"
    m.save(filename)
    print(f"Map for {pretty_names[var]} saved as {filename}")

print("All maps generated successfully!")


# ### Error Histograms (1x4)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Wider and lower
axes = axes.flatten()

final_df['Model'] = final_df['Model'].replace({
    "LSTM_13": "LSTM",
    "GeoLSTM_13": "GeoLSTM",
    "SARIMA_13": "SARIMA",
})

models_to_plot = ['LSTM', 'GeoLSTM', 'SARIMA']

residual_columns = [
    ('mean_speed_residuals', 50),
    ('percentile_10_residuals', 50),
    ('intTot_residuals', 4000),
    ('intP_residuals', 1000),
]

residual_titles = [
    'Mean Speed',
    '10th Percentile Speed',
    'Total Intensity',
    'HWV Intensity',
]

filtered_df = final_df[final_df['Model'].isin(models_to_plot)]

for i, ((res_col, x_lim), title) in enumerate(zip(residual_columns, residual_titles)):
    plot_combined_residuals_histogram(
        filtered_df,
        residual_column=res_col,
        model_column='Model',
        xaxis=x_lim,
        ax=axes[i])
    
    axes[i].set_title(title, fontsize=22)
    axes[i].set_xlabel('Residuals', fontsize=19)
    axes[i].set_ylabel('Density', fontsize=19)
    axes[i].legend().set_visible(False)

# Show legend only once
handles, labels = axes[0].get_legend_handles_labels()
axes[3].legend(
    handles, labels,
    title='Models',
    loc='upper right',
    fontsize=13,
    title_fontsize=13,
    frameon=True
)

plt.tight_layout()
plt.show()


# ### Error Scatterplot Geo
df_geo_vies = processing_geo_vies(folder_dades,file_geo_vies)
df_geo_vies=df_geo_vies.groupby(['pk']).agg({'ang_curv': 'max', 'ang_pend_pos':'max', 'ang_pend_neg':'max','segment':'max'}).reset_index()
df_geo_vies.pk = df_geo_vies.pk.astype(int)

# Load lon-lat data
df_lon_lat = pd.read_csv("lonlat_pks_ap7_120_220.csv", sep=",", decimal=".", encoding="latin1")
df_lon_lat['pk_rounded'] = np.trunc(df_lon_lat['pk'])
df_lon_lat = df_lon_lat.rename(columns={'pk':'pk_decimal'}).rename(columns={'pk_rounded':'pk'})
df_lon_lat = df_lon_lat[['pk', 'pk_decimal', 'lon', 'lat']].copy()
df_lon_lat.pk = df_lon_lat.pk.astype(int)
df_lon_lat = df_lon_lat.sort_values(by='pk_decimal')

# Merging the bdd with the road geometry
df_geo_vies = df_geo_vies.merge(df_lon_lat, on=['pk'], how='left')

model_name = "LSTM_13"

# Group and calculate MAE
mae_df = final_df[final_df['Model'] == model_name].copy()
mae_df=mae_df[['pk','mean_speed','std_dev_speed','percentile_10','percentile_85','intTot','intP','mean_speed_residuals_abs','std_dev_speed_residuals_abs','percentile_10_residuals_abs','percentile_85_residuals_abs','intTot_residuals_abs','intP_residuals_abs']].copy()
# Merge with lon-lat
df_geo_vies = df_geo_vies.merge(mae_df, on='pk')
del mae_df

# Define custom colormap: from pale blue to dark blue
colors = ["#d8ecf3", "#00008B"]  # LightBlue -> DarkBlue
custom_blue_cmap = LinearSegmentedColormap.from_list("pale_to_dark_blue", colors)

# Variables to plot
variables = [
    'mean_speed_residuals_abs', 
    'percentile_10_residuals_abs', 
    'intTot_residuals_abs', 
    'intP_residuals_abs'
]

titles = [
    'Mean Speed - LSTM',
    '10th Percentile - LSTM',
    'Total Intensity - LSTM',
    'Heavy-weight Intensity - LSTM'
]

# Y-axis limits
y_axis_vals = [
    ('mean_speed_residuals_abs', 120),
    ('percentile_10_residuals_abs', 120),
    ('intTot_residuals_abs', 7000),
    ('intP_residuals_abs', 1500),
]
y_limits = dict(y_axis_vals)

# Create figure with 4 subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
axes = axes.flatten()

for i, (var, title) in enumerate(zip(variables, titles)):
    ax = axes[i]
    
    sc = ax.scatter(
        df_geo_vies['ang_curv'], 
        df_geo_vies[var], 
        c=df_geo_vies[var], 
        cmap=custom_blue_cmap, 
        s=20,
        alpha=0.7
    )
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label(var, fontsize=8)
    
    # Axis settings
    ax.set_xlabel('Road Curvature', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True)
    ax.set_aspect('auto')
    
    # Set y-axis limit individually
    ax.set_ylim(0, y_limits[var])

plt.tight_layout()
plt.show()


df_geo_vies = processing_geo_vies(folder_dades,file_geo_vies)
df_geo_vies=df_geo_vies.groupby(['pk']).agg({'ang_curv': 'max', 'ang_pend_pos':'max', 'ang_pend_neg':'max','segment':'max'}).reset_index()
df_geo_vies.pk = df_geo_vies.pk.astype(int)

# Load lon-lat data
df_lon_lat = pd.read_csv("lonlat_pks_ap7_120_220.csv", sep=",", decimal=".", encoding="latin1")
df_lon_lat['pk_rounded'] = np.trunc(df_lon_lat['pk'])
df_lon_lat = df_lon_lat.rename(columns={'pk':'pk_decimal'}).rename(columns={'pk_rounded':'pk'})
df_lon_lat = df_lon_lat[['pk', 'pk_decimal', 'lon', 'lat']].copy()
df_lon_lat.pk = df_lon_lat.pk.astype(int)
df_lon_lat = df_lon_lat.sort_values(by='pk_decimal')

# Merging the bdd with the road geometry
df_geo_vies = df_geo_vies.merge(df_lon_lat, on=['pk'], how='left')

model_name = "GeoLSTM_13"

# Group and calculate MAE
mae_df = final_df[final_df['Model'] == model_name].copy()
mae_df=mae_df[['pk','mean_speed','std_dev_speed','percentile_10','percentile_85','intTot','intP','mean_speed_residuals_abs','std_dev_speed_residuals_abs','percentile_10_residuals_abs','percentile_85_residuals_abs','intTot_residuals_abs','intP_residuals_abs']].copy()
# Merge with lon-lat
df_geo_vies = df_geo_vies.merge(mae_df, on='pk')

# Define custom colormap: from pale blue to dark blue
colors = ["#d8ecf3", "#00008B"]  # LightBlue -> DarkBlue
custom_blue_cmap = LinearSegmentedColormap.from_list("pale_to_dark_blue", colors)

# Variables to plot
variables = [
    'mean_speed_residuals_abs', 
    'percentile_10_residuals_abs', 
    'intTot_residuals_abs', 
    'intP_residuals_abs'
]

y_axis_vals = [
    ('mean_speed_residuals_abs', 120),
    ('percentile_10_residuals_abs', 120),
    ('intTot_residuals_abs', 7000),
    ('intP_residuals_abs', 1500),
]

titles = [
    'Mean Speed - GeoLSTM',
    '10th Percentile GeoLSTM',
    'Total Intensity - GeoLSTM',
    'Heavy-weight Intensity - GeoLSTM'
]

# Create figure with 4 subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
axes = axes.flatten()

# Build a dictionary to easily retrieve y-limits
y_limits = dict(y_axis_vals)

for i, (var, title) in enumerate(zip(variables, titles)):
    ax = axes[i]
    
    sc = ax.scatter(
        df_geo_vies['ang_curv'], 
        df_geo_vies[var], 
        c=df_geo_vies[var], 
        cmap=custom_blue_cmap, 
        s=20,
        alpha=0.7
    )
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label(var, fontsize=8)
    
    # Axis settings
    ax.set_xlabel('Road Curvature', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True)
    ax.set_aspect('auto')
    
    # Set y-axis limit
    ax.set_ylim(0, y_limits[var])

plt.tight_layout()
plt.show()

# ### CFDs plots
# Create subplots
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes = axes.flatten()

# Define the residual columns and x-axis limits
residual_columns = [
    ('mean_speed_residuals', 50),
    ('percentile_10_residuals', 50),
    ('intTot_residuals', 8000),
    ('intP_residuals', 1500),
]

# Titles for each plot
residual_titles = [
    'Mean Speed',
    '10th Percentile Speed',
    'Total Intensity',
    'HWV Intensity'
]

# Updated operational thresholds
thresholds = {
    'mean_speed_residuals': 10,    # 10 km/h
    'percentile_10_residuals': 15, # 15 km/h
    'intTot_residuals': 1000,      # 1000 vehicles/hour
    'intP_residuals': 270    # 270 vehicles/hour
}

# Filter models to plot
models_to_plot = ['LSTM', 'GeoLSTM', 'SARIMA']
filtered_df=final_df.replace({"LSTM_13": "LSTM","GeoLSTM_13": "GeoLSTM","SARIMA_13": "SARIMA"})
filtered_df = filtered_df[filtered_df['Model'].isin(models_to_plot)]


# Plot CDFs
for i, ((res_col, x_lim), title) in enumerate(zip(residual_columns, residual_titles)):
    ax = axes[i]
    for model in models_to_plot:
        model_data = filtered_df[filtered_df['Model'] == model]
        residuals = np.abs(model_data[res_col].dropna())
        
        sorted_residuals = np.sort(residuals)
        cdf = np.arange(len(sorted_residuals)) / len(sorted_residuals)
        
        ax.plot(sorted_residuals, cdf, label=model, linewidth=2)
    
    # Add vertical line for operational threshold
    threshold = thresholds[res_col]
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Operational Threshold')
    
    ax.set_xlim([0, x_lim])
    ax.set_ylim([0, 1.0])
    ax.set_xlabel('Residuals', fontsize=19)
    ax.set_ylabel('Cumulative Probability', fontsize=19)
    ax.set_title(f'{title} - CDF', fontsize=22)
    ax.grid(True)
    if i == 2:  # Add legend only on the third plot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='Models', fontsize=17, title_fontsize=17, loc='lower right', frameon=True)

plt.tight_layout()
plt.show()


# ### GPU Power Usage
consumption_trials = []
for model in ['GeoLSTM','GeoTFT','GeoTransformer']:
    # Configuration
    power_column_name = 'GPU Power Usage (W)'  # Check that this matches your CSV columns exactly
    folder_path = os.path.join(os.path.dirname(os.path.join(os.getcwd())), 'results', 'gpu_power_usage', model)
    print(folder_path)
    seconds_between_measurements = 2  # Adjust if needed (e.g., if W&B logs every 2 seconds)

    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # Store results
    
    trial = 0

    for file in csv_files:
        # Load CSV
        df = pd.read_csv(file)
        print(df)
        # Compute differences and energy consumption
        df['diff_time'] = df['Relative Time (Process)'].shift(-1) - df['Relative Time (Process)']
        df['energy_consumption'] = (df['diff_time'] * df[df.columns[1]]) / (3600)
        df['energy_consumption'] = df['energy_consumption']

        total_energy = df['energy_consumption'].sum().round(decimals=4)
        mean_power = df[df.columns[1]].mean().round(decimals=4)
        time_duration = df['diff_time'].sum().round(decimals=0)

        
        # Save trial result
        consumption_trials.append({'model':model,'trial_num': trial, 'duration_min': time_duration/60, 'energy_consumption_Wh': total_energy, 'mean_power_W': mean_power})
        trial += 1

# Build final DataFrame
summary_df = pd.DataFrame(consumption_trials)

# Save summary to CSV
summary_df.to_csv('power_usage_summary.csv', index=False)
print("Summary saved to 'power_usage_summary.csv'")