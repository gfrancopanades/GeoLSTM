import os

data_version='preMARIA_v2_geo-cal-vel-int-ret_v1'
model_version='preMARIA_v2_cross-val_v1'

# Specify the path to your CSV folder
# Get the directory of the current script
folder_path=os.path.dirname(os.getcwd())

folder_velocitats=folder_dades=os.path.join(folder_path, 'resources','velocitats_pk_2022_2023')
folder_dades=os.path.join(folder_path, 'resources')
folder_visualizations=os.path.join(folder_path, 'visualizations')
folder_model=os.path.join(folder_path, 'model_'+model_version)

model_save_path = os.path.join(folder_model, f'model_{model_version}.h5')
history_save_path = os.path.join(folder_model, f'history_model_{model_version}.csv')
file_path_scaler = os.path.join(folder_model, f'scaler_{model_version}.csv')
# Specify the path to your CSV file
file_name_accidents = 'accidents_2022_2023_clean_filtered.csv'
file_name_sancions22 = 'sancions2022.csv'
file_name_sancions23 = 'sancions2023.csv'
file_name_retencions22 = 'retencions2022.csv'
file_name_retencions23 = 'retencions2023.csv'

file_name_retencionspks_22_23='retencionsPks2022_2023.csv'
file_name_mobilitat22 = 'mobilitat2022.csv'
file_name_mobilitat23 = 'mobilitat2023.csv'
file_name_mitma22='mobilitat_mitma_2022.csv'
file_name_mitma23='mobilitat_mitma_2023.csv'
file_name_velocitats_1='velocitats_juny23_oct23.csv'
file_name_velocitats_2='vels_gen23_maig23.csv'
file_name_velocitats_3='vels_juny22_dec22.csv'

file_accidents_treated="df_accidents_unif.csv"
file_retencions_treated="df_retencions_unif.csv"
file_mitma_treated="df_mitma_unif.csv"
file_velocitats_treated="df_velocitats_unif.csv"

file_name_mob_clean = 'mobilitat_mitma_AP7_120_220_2022_2023_wv_pk_treated_2.csv'
file_name_mob_clean_2 = 'mobilitat_mitma_AP7_120_220_2022_2023_wv_pk_no_treated.csv'
file_mobilitat_pk='mobilitat_2022_2024_pk_treated.csv'
file_retencions_pk='retencions_AP7_120_220_2022_2023.csv'
file_velocitats_pk='vels_AP7_120_220_gen22_dec23.csv'

pkini_pkfi_etds='pkini_pkfi_etds_unique.csv'

file_geo_vies='pksCur_AP7_120_220.csv'
file_cal_mob='dataFest_2021_2024.csv'

file_merge_total='final_data_treatment.csv'

file_meteo_barcelona='meteo_barcelona_22_23.csv'
file_meteo_sta_perpetua='meteo_santa_perpetua_mogoda_22_23.csv'
file_meteo_sabadell='meteo_sabadell_bellaterra_22_23.csv'

file_name_pk_etds="pkini_pkfi_etds.csv"


print(f">>> Reading and saving the CSV file into a DataFrames...")