#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data integration and preprocessing pipeline for hourly velocity and intensity features
along the AP-7 highway corridor.

This script constructs a complete spatiotemporal dataset from multiple sources:
velocity records, traffic intensity (MITMA), special mobility calendar, and
road geometry features. The result is a structured dataset ready for training
spatiotemporal predictive models.

Author: [Your Name]
Affiliation: Universitat Politècnica de Catalunya
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from src.data_file_names import *  # Path configuration
from src.data.data_utils import *  # Custom data treatment functions

# Disable MKL optimizations to prevent potential TensorFlow issues (if used downstream)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ensure output folder exists
check_and_create_folder(folder_model)

print("=" * 70)
print(">> Spatiotemporal data preprocessing: HOURLY AGGREGATION")
print("=" * 70)

# ------------------------------------------------------------------------------
# Step 1: Construct a full hourly spatiotemporal grid for the AP-7 corridor
# ------------------------------------------------------------------------------

start_date = '2022-01-11'
end_date = '2024-01-01'
via_values = ['AP-7']
pk_values = range(120, 220)
sen_values = ['dec']

date_range = pd.date_range(start=start_date, end=end_date, freq='1h')
grid = list(product(date_range, via_values, pk_values, sen_values))

df_base = pd.DataFrame(grid, columns=['dat', 'via', 'pk', 'sen'])
df_base['Any'] = df_base['dat'].dt.year
df_base['mes'] = df_base['dat'].dt.month
df_base['dia'] = df_base['dat'].dt.day
df_base['diaSem'] = df_base['dat'].dt.weekday
df_base['hor'] = df_base['dat'].dt.hour

print(f"> Spatiotemporal base created: {df_base.shape[0]} rows")

# ------------------------------------------------------------------------------
# Step 2: Merge with hourly velocity data
# ------------------------------------------------------------------------------

df_vel = pd.read_csv(os.path.join(folder_dades, "vels_120_220_jan22_dec23_hourly.csv"),
                     sep=",", decimal=".", encoding='latin-1')

df_vel = df_vel.rename(columns={'senCD': 'sen', 'any': 'Any', 'pk_rounded': 'pk'})
df_vel['via'] = 'AP-7'

# Aggregate by temporal and spatial keys
df_vel = df_vel.groupby(['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen']).agg({
    'mean_speed': 'mean',
    'std_dev_speed': 'mean',
    'percentile_10': 'mean',
    'percentile_85': 'mean'
}).reset_index()

df_merge = df_base.merge(df_vel, on=['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen'], how='left')
del df_base, df_vel

# ------------------------------------------------------------------------------
# Step 3: Merge with MITMA traffic intensity data
# ------------------------------------------------------------------------------

df_int = processing_intensitats_mitma(folder_dades, file_name_mob_clean_2, pkini_pkfi_etds)
df_int = df_int.groupby(['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen']).agg({
    'intTot': 'sum', 'intP': 'sum', 'car': 'max'
}).reset_index()

df_merge = df_merge.merge(df_int, on=['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen'], how='left')
del df_int

# ------------------------------------------------------------------------------
# Step 4: Merge with mobility calendar (e.g., holidays or high traffic)
# ------------------------------------------------------------------------------

df_cal = processing_cal_mob(folder_dades, file_cal_mob)
df_cal = df_cal.groupby(['Any', 'mes', 'dia']).agg({'mob_esp': 'max'}).reset_index()

df_merge = df_merge.merge(df_cal, on=['Any', 'mes', 'dia'], how='left')
df_merge['mob_esp'] = df_merge['mob_esp'].fillna(0)
del df_cal

# ------------------------------------------------------------------------------
# Step 5: Merge with road geometry data
# ------------------------------------------------------------------------------

df_geo = processing_geo_vies(folder_dades, file_geo_vies)
df_geo = df_geo.groupby(['via', 'pk', 'sen']).agg({
    'ang_curv': 'max',
    'ang_pend_pos': 'max',
    'ang_pend_neg': 'max',
    'segment': 'max'
}).reset_index()

df_merge = df_merge.merge(df_geo, on=['via', 'pk', 'sen'], how='left')
del df_geo

# ------------------------------------------------------------------------------
# Step 6: Mark imputed values and interpolate missing features
# ------------------------------------------------------------------------------

speed_cols = ['mean_speed', 'std_dev_speed', 'percentile_10', 'percentile_85']
intensity_cols = ['car', 'intTot', 'intP']

df_merge['speed_imputation'] = df_merge[speed_cols].isna().any(axis=1).astype(int)
df_merge['intensity_imputation'] = df_merge[intensity_cols].isna().any(axis=1).astype(int)

df_merge[speed_cols] = df_merge[speed_cols].interpolate(method='linear', limit_direction='both')
df_merge[intensity_cols] = df_merge[intensity_cols].interpolate(method='linear', limit_direction='both')

# ------------------------------------------------------------------------------
# Step 7: Standardize types and round numerical precision
# ------------------------------------------------------------------------------

columns_operations = {
    'mean_speed': (int, 0),
    'std_dev_speed': (float, 4),
    'percentile_10': (int, 0),
    'percentile_85': (int, 0),
    'intTot': (int, None),
    'intP': (int, None),
    'speed_imputation': (int, None),
    'intensity_imputation': (int, None),
    'ang_curv': (float, 2),
    'ang_pend_pos': (float, 2),
    'ang_pend_neg': (float, 2)
}

for column, (dtype, decimals) in columns_operations.items():
    if decimals is not None:
        df_merge[column] = df_merge[column].astype(dtype).round(decimals=decimals)
    else:
        df_merge[column] = df_merge[column].astype(dtype)

# ------------------------------------------------------------------------------
# Step 8: Save final dataset to CSV
# ------------------------------------------------------------------------------

output_path = os.path.join(folder_dades, f"{data_version}_vel_int_cal_mob_1h_new_etds_5.csv")
df_merge.to_csv(output_path, sep=";", decimal=".", encoding="latin-1", index=False)

print(f"> Data exported to: {output_path}")
print("Final DataFrame shape:", df_merge.shape)
print("Missing values per column:")
print(df_merge.isna().sum())
print("Duplicate rows:", df_merge.duplicated().sum())
