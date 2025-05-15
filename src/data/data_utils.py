import pandas as pd
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/src')

from src.data_file_names import *

def check_and_create_folder(path):
    # Check if the path exists
    if not os.path.exists(path):
        # Create the folder if it doesn't exist
        os.makedirs(path)
        print(f"Folder created at: {path}")
    else:
        print(f"Folder already exists at: {path}")

def processing_accidents(folder_dades, file_name_accidents, df_vels):
    print(f">>> Importing accident data...")
    df_accidents=pd.read_csv(os.path.join(folder_dades,file_name_accidents), sep=";", decimal=".")

    print(f">>> Formating date to datetime")
    df_accidents['dat'] = pd.to_datetime(df_accidents['dat'], format='%d/%m/%Y %H:%M')
    df_accidents['dat'] = df_accidents.dat.dt.round('5min')
    df_accidents['Any']=df_accidents.dat.dt.year
    df_accidents['mes']=df_accidents.dat.dt.month
    df_accidents['dia']=df_accidents.dat.dt.day
    df_accidents['hor']=df_accidents.dat.dt.hour
    df_accidents['5min']=df_accidents.dat.dt.minute
    df_accidents.pk=df_accidents.pk.astype(int)
    # Altres variables 'D_SENTITS_VIA', 'D_SENTIT_VEHICLES'
    df_accidents=df_accidents.groupby(['Any', 'mes', 'dia', 'hor', '5min', 'dat', 'via', 'pk', 'sen'])['accidents'].sum()
    df_vel_sent_acc=df_vels[['Any', 'mes', 'dia', 'hor', '5min', 'via', 'pk', 'distTramKm', 'mitjPonVelKm', 'mitjPonVelPlaca', 'mitjPonVelPat', 'mitjPonVelFF']].copy()
    print(f">>> Applying lambda function to assign way direction to each accident...")
    df_accidents['sen']=df_accidents.apply(lambda row: asign_slower_way(row,df_vel_sent_acc), axis=1)
    df_accidents

    return df_accidents

def asign_slower_way(row,df_vel_sent_acc):
  cond1 = (df_vel_sent_acc['Any']==row['Any'])
  cond2 = (df_vel_sent_acc['mes']==row['mes'])
  cond3 = (df_vel_sent_acc['dia']==row['dia'])
  cond4 = (df_vel_sent_acc['hor']==row['hor'])
  cond5 = (df_vel_sent_acc['via']==row['via'])
  cond6 = (df_vel_sent_acc['pk']==row['pk'])
  cond7 = (df_vel_sent_acc['min']==row['min'])
  combined_condition = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7
  df_way=df_vel_sent_acc[combined_condition].sort_values(by='mitjPonVelKm').reset_index(drop=True)
  try:
    return df_way.loc[0, 'sen']
  except:
    return 'unkwn'


def round_to_nearest_power_of_2(number):
    if number < 32:
        return 32

    powers_of_2 = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    for i in range(len(powers_of_2) - 1):
        lower = powers_of_2[i]
        upper = powers_of_2[i + 1]
        if lower <= number < upper:
            return lower if (number - lower) < (upper - number) else upper

    return powers_of_2[-1]

def processing_intensitats_depurades(folder_dades,file_mobilitat_pk_2):
    
    df=pd.read_csv(os.path.join(folder_dades,file_mobilitat_pk), sep=",", decimal=".", encoding='latin-1').rename(columns={'senCD':'sen'})
    print("Number of duplicate rows from imported file:", df.duplicated(subset=['Any','mes','dia','hor','via','pk','sen']).sum())
    
    df=df[['Any','mes','dia','hor','via','pk','sen','int']].reset_index(drop=True)
    # df=df.rename(columns={'int':'int_depurades'})
    # Checking for NaN values in 'int' and 'vel' columns
    print("Number of Nan rows from imported file:",df['int_depurades'].isna().sum())
    return df

def processing_intensitats_mitma(folder_dades, file_name_mob_clean_2, pkini_pkfi_etds):
    """
    Process intensity data from raw CSV files and prepare a clean, structured DataFrame.

    Parameters:
        folder_dades (str): Folder containing the input files.
        file_name_mob_clean_2 (str): File name for mobility intensity data.
        pkini_pkfi_etds (str): File name for initial and final PK ranges with ETDs.

    Returns:
        pd.DataFrame: Processed DataFrame with relevant columns and filtering applied.
    """
    # Load the initial data
    df = pd.read_csv(os.path.join(folder_dades, file_name_mob_clean_2), sep=",", decimal=".", encoding="latin-1")

    # Check and handle duplicates
    print("Number of duplicate rows from imported file:", df.duplicated(subset=['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen','car']).sum())
    df=df.drop_duplicates().reset_index(drop=True)
    print("Number of duplicate rows from imported file:", df.duplicated(subset=['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen','car']).sum())

    # Fill missing 'car' values and normalize 'sen' and 'via' column values
    df['sen'] = df['sen'].replace(1, "cre").replace(2, "dec")
    df['via'] = df['via'].replace({"AP-7N": "AP-7", "AP-7S": "AP-7", "AP7": "AP-7"})

    # Keep relevant columns and rename
    df = df[['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen', 'car', 'intL', 'intP', 'int']].rename(columns={'int': 'intTot'})

    ##########################
    value_counts = df['car'].value_counts()
    print(value_counts)
    num_nans = df['car'].isna().sum()
    print(num_nans)
    ##########################

    # Group by specific columns and sum numeric data
    df = df.groupby(['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen', 'car'], as_index=False).sum(numeric_only=True)

    # Compute maximum 'car' per group and merge with summed intensities
    df_num_car = df.groupby(['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen'], as_index=False)['car'].count()
    df = df.groupby(['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen'], as_index=False)[['intL', 'intP', 'intTot']].sum()
    df = df.merge(df_num_car, on=['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen'], how='left')

    # Filter and sort data
    print("Number of duplicate rows:", df.duplicated(subset=['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'car', 'sen']).sum())
    df = df[['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen', 'car', 'intTot', 'intP']].reset_index(drop=True)
    df.sort_values(by=['Any', 'mes', 'dia', 'hor', 'via', 'sen', 'pk', 'car'], inplace=True)
    df = df[df['sen'] == 'dec']

    # Load ETD ranges and map PK values
    df_etds = pd.read_csv(os.path.join(folder_dades, pkini_pkfi_etds), sep=";", decimal=".", encoding="latin-1")
    interval_index = pd.IntervalIndex.from_arrays(df_etds['pkIni'], df_etds['pkFi'], closed='both')
    df['ETD'] = df['pk'].map(lambda x: df_etds['ETD'][interval_index.contains(x)].iloc[0] if any(interval_index.contains(x)) else None)

    # Merge ETD data and filter ranges
    df = pd.merge(df, df_etds, on=['ETD'], how='left')
    df = df[(df['pkIni'] >= float(df_etds['pkIni'].min())) & (df['pkFi'] <= float(df_etds['pkFi'].max()))]
    df = df[(df['pk'] >= 120) & (df['pk'] <= 220)].sort_values(by='pk')

    # Explode PK ranges into individual rows
    pk_ranges = df.apply(lambda row: list(range(int(np.ceil(row['pkIni'])), int(np.floor(row['pkFi']) + 1))), axis=1)
    df = df.loc[df.index.repeat(pk_ranges.str.len())]
    df['pk'] = np.concatenate(pk_ranges.values)

    # Final sorting and column selection
    df = df.sort_values(by='pk').reset_index(drop=True)
    df = df[(df['pk'] >= 120) & (df['pk'] <= 220)].sort_values(by='pk')
    df = df[['Any', 'mes', 'dia', 'hor', 'via', 'pk', 'sen', 'car', 'intTot', 'intP']].reset_index(drop=True)

    return df

def tram_to_pk(df):
    # Initialize an empty list to store the expanded rows
    expanded_rows = []
    
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Generate integer pk values within the range [pkIni, pkFi], rounded to the nearest integer
        pk_range = range(int(np.ceil(row['pkIni'])), int(np.floor(row['pkFi']) + 1))
        
        # Append a new row for each pk value
        for pk in pk_range:
            # Create a duplicate row with the pk value
            new_row = row.copy()
            new_row['pk'] = pk
            expanded_rows.append(new_row)
    
    # Convert the expanded rows list into a new DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Optionally sort by the pk column if needed
    expanded_df = expanded_df.sort_values(by='pk').reset_index(drop=True)
    
    return expanded_df


def processing_retencions(folder_dades,file_retencions_pk):
    print(f"> Processing retentions BDD...")
    start_time = time.time()

    df = pd.read_csv(os.path.join(folder_dades,file_retencions_pk), sep=",", decimal=".", encoding='latin-1')
    df = df[['C_ID_AFECTACIO','Any', 'mes', 'dia', 'pk', 'horaMinut', 'via', 'sen', 'dataIni','DESC_CAUSA', 'C_NIVELL_AFECTACIO','DESCRIPCIO', 'F_TEMPS_AFECTACIO', 'F_LONG_AFECTACIO','F_FACTOR_RETENCIO']].copy()

    # Correcció de sentits (divisió Creixent i Decreixen)
    aux=df[df.sen=='Ambdós sentits'].copy()
    aux['sen']='Nord'
    df = pd.concat([df, aux], ignore_index=True)
    aux['sen']='Sud'
    df = pd.concat([df, aux], ignore_index=True)
    df['sen']=df['sen'].replace("Oest","Sud").replace("Sud","cre").replace("Nord","dec").replace("Sense especificar","dec")


    df=df[~(df.sen=='Ambdós sentits')].copy()

    df.horaMinut=pd.to_datetime(df.horaMinut)
    df['hor']=df.horaMinut.dt.hour
    df['min']=df.horaMinut.dt.minute

    df=df[['C_ID_AFECTACIO','Any','mes','dia','hor','min','pk','via','sen','dataIni','DESC_CAUSA','C_NIVELL_AFECTACIO','DESCRIPCIO','F_TEMPS_AFECTACIO','F_LONG_AFECTACIO','F_FACTOR_RETENCIO']].copy()
    df['dataIni'] = pd.to_datetime(df['dataIni'], errors='coerce')

    # Convert datetime objects to seconds since Unix epoch (1970-01-01)
    df['seconds'] = (df['dataIni'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

    # df = df.dropna(subset=['dataIni'])
    df=df.sort_values('dataIni', ascending=False).groupby(['Any', 'mes', 'dia', 'hor', 'min', 'pk', 'via', 'sen']).head(1)

    df['1_temps_transit_fluit']=0
    df['2_temps_transit_pesat']=0
    df['3_temps_transit_lent']=0
    df['4_temps_cua_en_moviment']=0
    df['5_temps_carretera_tallada']=0
    df['1_long_transit_fluit']=0
    df['2_long_transit_pesat']=0
    df['3_long_transit_lent']=0
    df['4_long_cua_en_moviment']=0
    df['5_long_carretera_tallada']=0

    df.loc[df.C_NIVELL_AFECTACIO==2, '2_temps_transit_pesat'] = 1
    df.loc[df.C_NIVELL_AFECTACIO==3, '3_temps_transit_lent'] = 1
    df.loc[df.C_NIVELL_AFECTACIO==4, '4_temps_cua_en_moviment'] = 1
    df.loc[df.C_NIVELL_AFECTACIO==5, '5_temps_carretera_tallada'] = 1

    df.loc[df.C_NIVELL_AFECTACIO==2, '2_long_transit_pesat'] = df.loc[df.C_NIVELL_AFECTACIO==2, 'F_LONG_AFECTACIO'].astype(int)
    df.loc[df.C_NIVELL_AFECTACIO==3, '3_long_transit_lent'] = df.loc[df.C_NIVELL_AFECTACIO==3, 'F_LONG_AFECTACIO'].astype(int)
    df.loc[df.C_NIVELL_AFECTACIO==4, '4_long_cua_en_moviment'] = df.loc[df.C_NIVELL_AFECTACIO==4, 'F_LONG_AFECTACIO'].astype(int)
    df.loc[df.C_NIVELL_AFECTACIO==5, '5_long_carretera_tallada'] = df.loc[df.C_NIVELL_AFECTACIO==5, 'F_LONG_AFECTACIO'].astype(int)

    df=df.sort_values(by='dataIni').reset_index(drop=False).copy()
    df['5min'] = df['min'].apply(lambda x: (x // 5) * 5)
    df=df.groupby(['Any','mes','dia','hor','5min','pk','via','sen']).agg({'2_temps_transit_pesat':'sum','3_temps_transit_lent':'sum','4_temps_cua_en_moviment':'sum','5_temps_carretera_tallada':'sum','2_long_transit_pesat':'mean','3_long_transit_lent':'mean','4_long_cua_en_moviment':'mean','5_long_carretera_tallada':'mean'}).reset_index(drop=False).copy()
    # pk_retencions_mask = (df.pk <= 169) & (df.pk >= 128)
    # df = df[pk_retencions_mask].reset_index(drop=False).copy()
    df = df[(df.pk <= 169) & (df.pk >= 128)].reset_index(drop=False).copy()
    # df=df[df.pk<=169][df.pk>=128].reset_index(drop=False).copy()
    df['1_temps_transit_fluit'] = 5 - df['2_temps_transit_pesat'] - df['3_temps_transit_lent'] - df['4_temps_cua_en_moviment'] - df['5_temps_carretera_tallada']
    del df['index']

    # Creation of the new variable 1_temps_transit_fluit
    cond1 = df['2_temps_transit_pesat'].isna()
    cond2 = df['3_temps_transit_lent'].isna()
    cond3 = df['4_temps_cua_en_moviment'].isna()
    cond4 = df['5_temps_carretera_tallada'].isna()
    cond_total = cond1 & cond2 & cond3 & cond4
    df.loc[cond_total, '1_temps_transit_fluit'] = 5
    df = df.fillna(0)
    df.sort_values(by=['Any','mes','dia','hor','via','sen','pk'],inplace=True)

    exec_time=time.time() - start_time
    print(f"Execution ended successfully! Duration: {exec_time:.2f} seconds")
    print("======================================================================")
    return df


def processing_velocitats_azure(folder_dades,file_velocitats_pk):
    print(f"> Processing speed BDD...")
    start_time = time.time()
    df = pd.read_csv(os.path.join(folder_dades,file_velocitats_pk), sep=",", decimal=".", encoding='latin-1')
    return df


def processing_velocitats_civicat(folder_dades,file_velocitats_pk, folder_velocitats):
    print(f"> Processing speed BDD...")
    start_time = time.time()

    # List all CSV files in the folder manually and concatenate all files into one DataFrame
    all_files = [os.path.join(folder_velocitats, f) for f in os.listdir(folder_velocitats) if f.endswith('.csv')]
    df = pd.concat((pd.read_csv(file, sep=",", decimal=".", encoding='latin-1') for file in all_files), ignore_index=True)
    print(f">>> Total files concatenated: {len(all_files)}")

    # df = pd.read_csv(os.path.join(folder_dades,file_velocitats_pk), sep=",", decimal=".", encoding='latin-1')
    df=df[['via','sen', 'pkIni', 'pkFi', 'pkMin', 'pkMax', 'velPlaca', 'bearing','senCIVICAT', 'agrTem', 'dat', 'vel','velPat', 'velFF', 'trMinuts', 'score', 'cvalue', 'err']].copy()

    # Compute the distance of each original section and the distance x speed column
    print(f">>> Computing the distance of each original section and the distance x speed column...")
    df['distancia']=df.pkMax-df.pkMin
    df['pkMinTrunc']=df.pkMin.astype(int)
    df['mitjPonVelKm']=df['distancia']*df['vel']
    df['mitjPonVelPlaca']=df['distancia']*df['velPlaca']
    df['mitjPonVelPat']=df['distancia']*df['velPat']
    df['mitjPonVelFF']=df['distancia']*df['velFF']

    # Groupby the distance x speed column for each speed type
    print(f">>> Grouping by the distance x speed column for each speed type...")
    df=df.groupby(['via','sen','pkMinTrunc','dat'])[['distancia','mitjPonVelKm','mitjPonVelPlaca','mitjPonVelPat','mitjPonVelFF']].sum().reset_index(drop=False)
    df=df.rename(columns={'distancia':'distTramKm','pkMinTrunc':'pk'})

    # Divide the distance x speed column by the total distance of the whole kilometric section
    print(f">>> Dividing the distance x speed column by the total distance of the whole kilometric section...")
    df['mitjPonVelKm']=(df['mitjPonVelKm']/df['distTramKm']).round(decimals=2)
    df['mitjPonVelPlaca']=(df['mitjPonVelPlaca']/df['distTramKm']).round(decimals=2)
    df['mitjPonVelPat']=(df['mitjPonVelPat']/df['distTramKm']).round(decimals=2)
    df['mitjPonVelFF']=(df['mitjPonVelFF']/df['distTramKm']).round(decimals=2)

    # Convert the date column to datetime to divide it in 4 columns
    print(f">>> Converting the date column to datetime to divide it in 4 columns...")
    df.dat=pd.to_datetime(df.dat)
    df['hor']=df.dat.dt.hour
    df['Any']=df.dat.dt.year
    df['mes']=df.dat.dt.month
    df['dia']=df.dat.dt.day
    df['5min']=df.dat.dt.minute

    df=df[['Any','mes','dia','hor','5min','pk','via','sen','distTramKm','mitjPonVelKm','mitjPonVelPlaca','mitjPonVelPat','mitjPonVelFF']].reset_index(drop=False)
    # df=df[df.pk<=169][df.pk>=128].reset_index(drop=False)
    del df['index']
    del df['distTramKm']
    del df['mitjPonVelPat']
    del df['mitjPonVelFF']
    del df['mitjPonVelPlaca']
    exec_time=time.time() - start_time
    print(f"Execution ended successfully! Duration: {exec_time:.2f} seconds")
    print("======================================================================")

    return df

def get_first_decimal(column):
    """
    Extracts the first decimal digit from a numeric column.
    
    Parameters:
        column (pd.Series): The numeric column from which to extract the first decimal.
    
    Returns:
        pd.Series: A new column with the first decimal digit extracted.
    """
    return ((column * 10) % 10).astype(int)

def processing_geo_vies(folder_dades,file_geo_vies):
    """
    -------- PLOTTING THE HISTOGRAM TO DECIDE THE CURVE CATEGORY THRESHOLD -------
    import matplotlib.pyplot as plt

    # Select a column, for example 'mean_speed'
    column_name = 'radius'

    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        # Drop NaN values before plotting
        data = df[column_name][df[column_name]<10000][df[column_name]>-10000].dropna()
        
        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=1000, edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")
    """
    print(f"> Importing road geometry BDD...")
    start_time = time.time()

    df = pd.read_csv(os.path.join(folder_dades, file_geo_vies), sep=",", decimal=".", encoding='latin-1')

    # Melt the DataFrame to restructure slope data
    df = df.melt(id_vars=['via', 'pk', 'x', 'y', 'Z', 'rad'], value_vars=['mAsc', 'mDes'], var_name='sen', value_name='slope')

    df['ang_curv'] = np.degrees(np.arctan(100 / df['rad'])).round(decimals=2).abs()
    df['ang_pend'] = np.degrees(np.arcsin(df['slope'] / 100)).round(decimals=2)
    df['ang_pend_pos'] = 0
    df['ang_pend_neg'] = 0
    df['ang_pend_pos'] = df.loc[df['ang_pend']>=0,'ang_pend']
    df['ang_pend_neg'] = df.loc[df['ang_pend']<0,'ang_pend']


    # Store original curvature and slope values
    df['orig_radius'] = df['rad']
    df['orig_slope'] = df['slope']

    # Add a segment number (0-9) to track original positions within each PK
    df['segment'] = get_first_decimal(df['pk'])

    # Modify 'sen' to categorical names
    df['sen'] = df['sen'].replace({'mAsc': 'cre', 'mDes': 'dec'})

    # Round PK values for aggregation
    df['pk_rounded'] = df['pk'].astype(float).round(0).astype(int)

    # Compute curvature cumulative measure
    df['curve_cum'] = (2 * math.pi) / df['rad'].abs()

    # Initialize curve categories
    df['curve_light'] = 0
    df['curve_regular'] = 0
    df['curve_heavy'] = 0

    # Assign curvature categories
    df.loc[df['orig_radius'].abs() <= 300, 'curve_heavy'] = 1
    df.loc[(df['orig_radius'].abs() > 300) & (df['orig_radius'].abs() <= 1300), 'curve_regular'] = 1
    df.loc[(df['orig_radius'].abs() > 1300), 'curve_light'] = 1

    # Assign slope values for aggregation
    df['slope_rise'] = df['slope']
    df['slope_drop'] = df['slope']
    df['slope_cum'] = df['slope'].abs()


    # Aggregate per PK while keeping original values in lists
    df_agg = df.groupby(['via', 'pk_rounded', 'sen']).agg({
        'slope_cum': 'sum',
        'slope_rise': 'max',
        'slope_drop': 'min',
        'curve_light': 'sum',
        'curve_regular': 'sum',
        'curve_heavy': 'sum',
        'curve_cum': 'sum',
        'ang_curv': 'sum',
        'ang_pend_pos': 'sum',
        'ang_pend_neg': 'sum',
        'orig_radius': lambda x: list(x),  # Store list of 10 values
        'orig_slope': lambda x: list(x)
    }).reset_index()

    # Count the number of original segments per PK (each PK should have 10)
    df_num = df.groupby(['via', 'pk_rounded', 'sen'], as_index=False)['segment'].nunique()
    # Merge this count back into df_agg
    df_agg = df_agg.merge(df_num[['via', 'pk_rounded', 'sen', 'segment']], on=['via', 'pk_rounded', 'sen'], how='left')

    # Divide curvature values by the number of original segments
    df_agg[['curve_light', 'curve_regular', 'curve_heavy']] = df_agg[['curve_light', 'curve_regular', 'curve_heavy']].div(df_agg['segment'], axis=0).round(decimals=2)

    # Convert slopes to absolute values
    df_agg['slope_drop'] = df_agg['slope_drop'].abs()
    df_agg['slope_rise'] = df_agg['slope_rise'].abs()
    df_agg['slope_cum'] = df_agg['slope_cum'].abs()

    # Ensure lists have exactly 10 elements per PK (pad with NaN if needed)
    df_agg['orig_radius'] = df_agg['orig_radius'].apply(lambda x: x[:10] if len(x) >= 10 else x + [float('nan')] * (10 - len(x)))
    df_agg['orig_slope'] = df_agg['orig_slope'].apply(lambda x: x[:10] if len(x) >= 10 else x + [float('nan')] * (10 - len(x)))

    # Convert lists to separate columns (pivoting)
    df_pivot = df_agg[['via', 'pk_rounded', 'sen']].copy()
    for i in range(10):
        df_pivot[f'orig_radius_{i}'] = df_agg['orig_radius'].apply(lambda x: x[i])
        df_pivot[f'orig_slope_{i}'] = df_agg['orig_slope'].apply(lambda x: x[i])

    # Merge with aggregated features
    df_final = df_agg.drop(columns=['orig_radius', 'orig_slope']).merge(df_pivot, on=['via', 'pk_rounded', 'sen'], how='left')

    # Rename PK column and return the processed DataFrame
    df_final = df_final.rename(columns={'pk_rounded': 'pk'})

    df_final = df_final.fillna(0)

    return df_final

def processing_cal_mob(folder_dades,file_cal_mob):
    print(f"> Importing special mobility calendar BDD...")
    start_time = time.time()

    df = pd.read_csv(os.path.join(folder_dades,file_cal_mob), sep=",", decimal=".", encoding='latin-1')
    
    df['dat']=pd.to_datetime(df['dat'])
    df['Any']=df.dat.dt.year
    df['mes']=df.dat.dt.month
    df['dia']=df.dat.dt.day
    df=df[['Any','mes','dia','senOpe','pont']].copy()
    df['mob_esp']=1
    df['senOpe']=df['senOpe'].fillna('ambdos')
    df = pd.get_dummies(df, columns=['senOpe', 'pont'])
    df = df.astype(int)
    df.drop_duplicates(inplace=True)

    exec_time=time.time() - start_time
    print(f"Execution ended successfully! Duration: {exec_time:.2f} seconds")
    print("======================================================================")

    return df



def processing_meteo(folder_dades,file_meteo, poblacio):
    print(f"> Importing meteo calendar BDD from {poblacio}...")
    start_time = time.time()

    df=pd.read_csv(os.path.join(folder_dades,file_meteo), sep=",", decimal=".", encoding='latin-1')
    
    df=df.rename(columns={'ï»¿date':'dat'}).fillna(0)
    df['dat']=pd.to_datetime(df['dat'])
    df['Any']=df.dat.dt.year
    df['mes']=df.dat.dt.month
    df['dia']=df.dat.dt.day
    df=df[['Any', 'mes', 'dia','tavg','tmin','tmax','prcp','snow','wdir','wspd','wpgt','pres','tsun']].copy()
    df[['Any', 'mes', 'dia']]=df[['Any', 'mes', 'dia']].astype(int)
    df[['tavg','tmin','tmax','prcp','snow','wdir','wspd','wpgt','pres','tsun']]=df[['tavg','tmin','tmax','prcp','snow','wdir','wspd','wpgt','pres','tsun']].astype(float)
    list_cols=list(df.columns)
    list_ignore=['dat','Any','mes','dia']
    list_cols = [item for item in list_cols if item not in list_ignore]
    for col in list_cols:
        df=df.rename(columns={col:col+'_'+poblacio})
    exec_time=time.time() - start_time
    print(f"Execution ended successfully! Duration: {exec_time:.2f} seconds")
    print("======================================================================")

    return df


def creating_empty_dataset(start_date, end_date, freq, via_values, pk_values, sen_values):
    print(f"> Creating empty dataset with the date and road range...")
    start_time = time.time()

    import pandas as pd
    from itertools import product

    # Define the ranges for each column
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Generate all combinations using itertools.product
    combinations = list(product(date_range, via_values, pk_values, sen_values))

    # Create a DataFrame with the combinations
    df_base = pd.DataFrame(combinations, columns=['dat', 'via', 'pk', 'sen'])

    df_base['Any']=df_base.dat.dt.year
    df_base['mes']=df_base.dat.dt.month
    df_base['dia']=df_base.dat.dt.day
    df_base['diaSem']=df_base.dat.dt.weekday
    df_base['hor']=df_base.dat.dt.hour
    df_base['5min']=df_base.dat.dt.minute

    df_base['via']=df_base['via'].astype(str)
    df_base['sen']=df_base['sen'].astype(str)
    df_base['pk']=df_base['pk'].astype(int)

    df_base=df_base[['dat','Any','mes','dia','diaSem','hor','5min','via','pk','sen']].copy()

    exec_time=time.time() - start_time
    print(f"Execution ended successfully! Duration: {exec_time:.2f} seconds")
    print("======================================================================")

    return df_base

def plot_heat_map_data(version,df_merge):
    df_plot_base=pd.DataFrame(df_merge.groupby(['Any','mes','dia'])['via'].count()).reset_index(drop=False).rename(columns={'via':'recompte_registres_info'}).pivot(index=['Any','mes'], columns='dia', values='recompte_registres_info').fillna(0)
    # Creating a heatmap using seaborn
    colors = sns.color_palette("YlGnBu")
    sns.heatmap(df_plot_base, annot=False, cmap=colors, fmt='.2f', cbar=True)
    plt.savefig(os.path.join(folder_visualizations,f'heatmap_data_plot_{version}.png'))
    plt.close() 