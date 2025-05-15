#!/usr/bin/env python
# coding: utf-8

# ### Libraries instalation

# pip install pandas==2.1.1
# pip install numpy==1.25.2
# pip install matplotlib==3.8.0
# pip install scikit-learn==1.3.1
# pip install seaborn==0.12.2
# pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu117
# pip install pytorch-lightning==2.1.2
# pip install optuna==3.3.0
# pip install torch-geometric


# ### Libraries import
import os
import time
import gc
import logging
import shutil

import itertools
import json
import joblib  # Optional for other configurations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data

# Custom functions from src
from src.data_file_names import *
from training.io import *
from src.data.data_utils import *
from src.training.utils import *


from datetime import timedelta, datetime

import optuna
import wandb


import torch
print(torch.cuda.get_device_properties(0))

import multiprocessing
print(f"multiprocessing.cpu_count() = {multiprocessing.cpu_count()}")


import time
import torch
from torch.utils.data import DataLoader

# ### Model Macro-parameters
model_time='1h'
model_vars='vels_ints_weig'
geo_vals=False
vw_val=True

# Utilitzant data_load_batch=64*16 funciona pero amb moltes epochs i trials es mor el kernel
# Initial batch size 1.024
# Final batch size 24.576

data_load_batch=64 #*16 # *6*4
datload_num_work=12

epochs = 100
trials= 30

iterat_num='15'
#delta_huber_loss=0.15

model_list_training_loop=['LSTMModel','TFTModel','TransformerModel']

pat_train=40
min_del_train=0.0001


import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_properties(0))


# ### Data treatment
# Initialize paths and directories
check_and_create_folder(folder_model)
check_and_create_folder(os.path.join(folder_path, 'dades'))

# Custom logger for logging events
class SimpleLogger(pl.Callback):
    def on_fit_end(self, trainer, pl_module):
        print(">>> Training completed successfully!")

# Load data
print("> Opening processed data...")
start_time = time.time()

df_merge = pd.read_csv(os.path.join(folder_dades, data_version + '_vel_int_cal_mob_1h_new_etds_5.csv'), sep=";", decimal=".", encoding='latin-1')
df_merge.sort_values(by=['via','sen','pk','Any','mes','dia','hor'],inplace=True)

cols_targets=['mean_speed', 'std_dev_speed', 'percentile_10', 'percentile_85','intTot', 'intP']
temp_features=['mob_esp','speed_imputation', 'intensity_imputation']
static_features=['car','segment','ang_curv','ang_pend_pos','ang_pend_neg']

curv_columns=['ang_curv']
slope_columns=['ang_pend_pos','ang_pend_neg']
imputation_columns=['speed_imputation','intensity_imputation']

extra_geo=['curve_light','curve_regular', 'curve_heavy', 'curve_cum', 'orig_radius_1',  'orig_radius_2', 'orig_radius_3', 'orig_radius_4', 'orig_radius_5', 'orig_radius_6', 'orig_radius_7', 'orig_radius_8',  'orig_radius_9','slope_cum', 'slope_rise', 'slope_drop', 'orig_radius_0',  'orig_slope_0',  'orig_slope_1', 'orig_slope_2', 'orig_slope_3', 'orig_slope_4', 'orig_slope_5', 'orig_slope_6', 'orig_slope_7', 'orig_slope_8',  'orig_slope_9']

# Define a dictionary with columns and their desired types and rounding decimals
columns_operations = {
    'mean_speed': (int, None),
    'std_dev_speed': (float, 4),
    'percentile_10': (int, None),
    'percentile_85': (int, None),
    'intTot': (int, None),
    'intP': (int, None),
    # 'curve_light': (int, None),
    # 'curve_regular': (int, None),
    # 'curve_heavy': (int, None),
    'speed_imputation': (int, None),
    'intensity_imputation': (int, None),
    # 'slope_cum': (float, 1),
    # 'slope_rise': (float, 1),
    # 'slope_drop': (float, 1),
    # 'curve_cum': (float, 6),
    # 'orig_radius_0':(float, 6),
    # 'orig_slope_0':(float, 6),
    # 'orig_radius_1':(float, 6),
    # 'orig_slope_1':(float, 6),
    # 'orig_radius_2':(float, 6),
    # 'orig_slope_2':(float, 6),
    # 'orig_radius_3':(float, 6),
    # 'orig_slope_3':(float, 6),
    # 'orig_radius_4':(float, 6),
    # 'orig_slope_4':(float, 6),
    # 'orig_radius_5':(float, 6),
    # 'orig_slope_5':(float, 6),
    # 'orig_radius_6':(float, 6),
    # 'orig_slope_6':(float, 6),
    # 'orig_radius_7':(float, 6),
    # 'orig_slope_7':(float, 6),
    # 'orig_radius_8':(float, 6),
    # 'orig_slope_8':(float, 6),
    # 'orig_radius_9':(float, 6),
    # 'orig_slope_9':(float, 6)
}

# Apply type conversion and rounding
for column, (dtype, decimals) in columns_operations.items():
    if decimals is not None:
        df_merge[column] = df_merge[column].astype(dtype).round(decimals=decimals)
    else:
        df_merge[column] = df_merge[column].astype(dtype)

if (geo_vals==False)&(vw_val==False):
    geo_vals_name="_NO_geo"
    vw_vals_name="_NO_vw"
    cols_targets=['mean_speed', 'std_dev_speed','percentile_10','percentile_85','intTot','intP']
    df_merge=df_merge.drop(columns=static_features)
elif (geo_vals==True)&(vw_val==False):
    geo_vals_name="_YES_geo"
    vw_vals_name="_NO_vw"
    cols_targets=['mean_speed', 'std_dev_speed','percentile_10','percentile_85','intTot','intP']

elif (geo_vals==False)&(vw_val==True):
    geo_vals_name="_NO_geo"
    vw_vals_name="_YES_vw"
    cols_targets=['mean_speed', 'std_dev_speed','percentile_10','percentile_85','intTot','intP']
    df_merge=df_merge.drop(columns=static_features)

else:
    geo_vals_name="_YES_geo"
    vw_vals_name="_YES_vw"
    columns = [col for col in df_merge.columns if col not in static_features] + static_features
    # df_merge=df_merge.drop(columns=extra_geo)

print(df_merge.columns)


df_merge_plot=df_merge.groupby('pk')[['dat']].count().reset_index()
# Plot
plt.figure(figsize=(10, 5))
plt.plot(df_merge_plot['pk'], df_merge_plot['dat'])
plt.xlabel('pk')
plt.ylabel('Count of mean_speed_residuals_abs')
plt.title('Line Plot of Count of mean_speed_residuals_abs by pk')
plt.grid(True)
plt.tight_layout()
plt.show()



# df_merge = df_merge.drop_duplicates()
unique_cols = ['dat', 'Any', 'mes', 'dia', 'diaSem', 'hor', 'via', 'pk', 'sen']
duplicates = df_merge[df_merge.duplicated(subset=unique_cols, keep=False)]
duplicates[['mean_speed', 'std_dev_speed', 'percentile_10', 'percentile_85','intTot', 'intP']].shape


print(f"Data columns: {df_merge.columns}")
print(f"Opening data completed! Duration: {time.time() - start_time:.2f} seconds")


print(f"Data columns: {df_merge.columns}")



# # Feature treatment
# df_merge['sen'] = df_merge['sen'].map({'cre': 1, 'dec': 0})
# df_merge['via'] = df_merge['via'].map({'AP-7': 0})

# df_merge['hor_sin'] = np.sin(2 * np.pi * df_merge['hor'] / 24)
# df_merge['hor_cos'] = np.cos(2 * np.pi * df_merge['hor'] / 24)

# df_merge['diaSem_sin'] = np.sin(2 * np.pi * df_merge['diaSem'] / 7)
# df_merge['diaSem_cos'] = np.cos(2 * np.pi * df_merge['diaSem'] / 7)

# df_merge['mes_sin'] = np.sin(2 * np.pi * df_merge['mes'] / 12)
# df_merge['mes_cos'] = np.cos(2 * np.pi * df_merge['mes'] / 12)


# If your dataset starts on January 1, 2023, you would define it like this:
initial_train_start_time = datetime(2022, 1, 1)
# Assuming you're training on the first 3 months of data, you would set it like this:
initial_train_end_time = datetime(2023, 9, 1)  # Example: March 31, 2023
# For instance, if your data ends on December 31, 2023:
end_of_data_time = datetime(2023, 12, 31)


if '1_temps_transit_fluit' in df_merge.columns:
    del df_merge['1_temps_transit_fluit']
    del df_merge['2_temps_transit_pesat']
    del df_merge['3_temps_transit_lent']
    del df_merge['4_temps_cua_en_moviment']
    del df_merge['5_temps_carretera_tallada']


# Feature treatment
df_merge['sen'] = df_merge['sen'].map({'cre': 1, 'dec': 0})
df_merge['via'] = df_merge['via'].map({'AP-7': 0})
df_merge_train_val=df_merge[(pd.to_datetime(df_merge['dat'])>=initial_train_start_time)&(pd.to_datetime(df_merge['dat'])<initial_train_end_time)]


print(df_merge_train_val[cols_targets].shape)
print(df_merge_train_val.drop(columns=cols_targets).drop(columns='dat').shape)


scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()
features_normalized = scaler_features.fit_transform(df_merge_train_val.drop(columns=cols_targets).drop(columns='dat'))
targets = scaler_targets.fit_transform(df_merge_train_val[cols_targets].copy())


# Prepare spatial features (last 7 columns of input_features)
spatial_feature_columns = df_merge_train_val.drop(columns=cols_targets).drop(columns='dat').columns[-7:]
temporal_feature_columns = df_merge_train_val.drop(columns=cols_targets).drop(columns='dat').columns[:-7]


# Features and targets for the simulation after the training
df_merge_sim = df_merge[(pd.to_datetime(df_merge['dat'])>=initial_train_end_time)&(pd.to_datetime(df_merge['dat'])<end_of_data_time)]


df_merge_train_val.drop(columns=cols_targets).drop(columns='dat').columns

# Assuming features_normalized and targets are numpy arrays or pandas DataFrames/Series
train_size = 0.8

# Calculate split indices
train_index = int(len(features_normalized) * train_size)

# Split the train data
features_train = features_normalized[:train_index]
targets_train = targets[:train_index:]

# Split the validation data
features_val = features_normalized[train_index:]
targets_val = targets[train_index:]


# ### Tensors and Dataloaders
input_size = features_train.shape[1] # Input size: 14
output_size = targets_train.shape[1] # Output size: 8
sequence_length = 1

print(f'Input size: {input_size}')
print(f'Output size: {output_size}')

WANDB_PROJECT = "MARIA-v1"
WANDB_ENTITY = "gfrancopanades-upc"  # Optional: Set your W&B account/organization name
os.environ["WANDB_API_KEY"] = "35467043207438848f5db049a454142b261a720a"
os.environ["WANDB_SILENT"] = "true"

total_elements = targets_train.size  # Total number of elements in targets_train
num_samples = total_elements // output_size  # Total samples

# Check potential sequence lengths
for seq_len in range(1, num_samples + 1):
    if num_samples % seq_len == 0:
        print(f"Compatible sequence length: {seq_len}")
        break  # Find the smallest compatible sequence length

# Convert to PyTorch tensors
targets_train_tensor = torch.tensor(targets_train, dtype=torch.float32).view(-1, sequence_length, output_size)
targets_val_tensor = torch.tensor(targets_val, dtype=torch.float32).view(-1, sequence_length, output_size)
features_train_tensor = torch.tensor(features_train, dtype=torch.float32).view(-1, sequence_length, input_size)
features_val_tensor = torch.tensor(features_val, dtype=torch.float32).view(-1, sequence_length, input_size)


# for workers in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
#     duration = benchmark_dataloader(workers, targets_train_tensor, data_load_batch)
#     print(f"num_workers={workers}, time taken: {duration:.3f} seconds")


# Create DataLoaders
train_loader = DataLoader(TensorDataset(features_train_tensor, targets_train_tensor), batch_size=data_load_batch, shuffle=False, num_workers=datload_num_work,persistent_workers=True, pin_memory=True)
val_loader = DataLoader(TensorDataset(features_val_tensor, targets_val_tensor), batch_size=data_load_batch, shuffle=False, num_workers=datload_num_work,persistent_workers=True,pin_memory=True)

for inputs, labels in val_loader:
    print(f"Batch Inputs shape: {inputs.shape}")
    print(f"Batch Labels shape: {labels.shape}")
    break  # Check just the first batch to verify shapes

# Models import
from src.models.attention_lstm import *
from src.models.conv_2_lstm import *
from src.models.conv_2_tft import *
from src.models.conv_2_transformer import *
from src.models.conv_lstm import *
from src.models.conv_tft import *
from src.models.conv_transformer import *
from src.models.lstm import *
from src.models.tft import *
from src.models.transformer import *
from src.models.x_lstm import *

# ### Model Training
# #### Functions
# Suppress PyTorch Lightning and torch logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Suppress LOCAL_RANK logs
os.environ["LOCAL_RANK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

iterat=iterat_num+geo_vals_name+vw_vals_name

model_mapping = {
    'LSTMModel': LSTMModel,
    'LSTMAttModel': LSTMAttModel,
    'xLSTMModel': xLSTMModel,
    'TransformerModel':TransformerModel,
    'Conv_LSTM_Model':ConvLSTMModel,
    'ConvTransformerModel':ConvTransformerModel,
    'ConvTFTModel':ConvTFTModel,
    'TFTModel':TFTModel,
    'DoubleConvTFTModel':DoubleConvTFTModel,
    'DoubleConvLSTM':DoubleConvLSTM,
    'DoubleConvTransformerModel':DoubleConvTransformerModel
    }

# #### Hyperparameter exploration loop
for model_name in model_list_training_loop:
    print(f'>>> PROCESSING MODEL {model_name}:')

    # Redirect stdout to a log file
    log_file = f'Training_log_model={model_name}_time={model_time}_iter{iterat}.txt'
    # Create the file if it doesn't exist or clear it if it does
    if not os.path.exists(log_file):
        open(log_file, "w").close()
    
    HYPERPARAMS = get_hyperparams(model_name)

    #########################################################################################################################
    #########################################################################################################################

    # Start the timer
    start_time = time.time()

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")  # 'minimize' for minimizing validation loss

    objective = create_objective(
        model_name=model_name,
        model_mapping=model_mapping,
        HYPERPARAMS=get_hyperparams(model_name),
        WANDB_PROJECT="your_project",
        WANDB_ENTITY="your_entity",
        input_size=64,
        output_size=3,
        train_loader=train_loader,
        val_loader=val_loader,
        static_features=static_features,
        curv_columns=curv_columns,
        slope_columns=slope_columns,
        log_file="training_log.txt",
        iterat="001",
        model_time="2025_05_14",
        pat_train=10,
        min_del_train=1e-4,
        epochs=50
    )

    study.optimize(objective, n_trials=50, n_jobs=1)

    # Stop the timer
    end_time = time.time()
    
    # Calculate the duration and print it
    duration = end_time - start_time
    print('=============================================================================')
    print('=============================================================================')

    print(f"Optuna optimization duration: {duration:.2f} seconds")

    # After optimization, print the best hyperparameters
    print("Best hyperparameters found by Optuna:")
    print(study.best_params)

    # Check the value of the best objective (e.g., best validation loss)
    print(f"Best validation loss: {study.best_value}")
    print('=============================================================================')
    print('=============================================================================')

    #########################################################################################################################
    #########################################################################################################################

    # Save directory for the trained model
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    hidden_size = study.best_params['hidden_size']  # Optimized hidden size
    num_layers = study.best_params['num_layers']  # Optimized number of layers
    dropout_prob = study.best_params['dropout_prob']  # Optimized dropout probability
    learning_rate = study.best_params['learning_rate']  # Optimized learning rate
    weight_decay = study.best_params['weight_decay']  # Optimized weight decay
    n_heads = study.best_params["n_heads"]
    dim_feedforward = study.best_params["dim_feedforward"]
    kernel_size = study.best_params["kernel_size"]
    stride = study.best_params["stride"]
    gcn_layers = study.best_params["gcn_layers"]
    gcn_hidden_size = study.best_params["gcn_hidden_size"]

    # Start the timer
    start_time = time.time()

    # Instantiate the model with suggested hyperparameters
    model_class = model_mapping[model_name]  # model_name should be set to one of the four model 

    if model_name in ['TransformerModel','TFTModel']:
        model = model_class( input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, n_heads=n_heads, dim_feedforward=dim_feedforward)
    elif model_name in ['LSTMAttModel', 'xLSTMAttModel']:
        model = model_class( input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, n_heads=n_heads)
    elif model_name == 'DoubleConvLSTM':
        model = model_class(input_size=input_size, hidden_size=hidden_size, num_spatial_features_group1=len(curv_columns), num_spatial_features_group2=len(slope_columns), num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, kernel_size=kernel_size, stride=stride)
    elif model_name == 'Conv_LSTM_Model':
        model = model_class(input_size=input_size, hidden_size=hidden_size, num_spatial_features=len(static_features), num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, kernel_size=kernel_size, stride=stride)
    elif model_name == 'GCN_LSTM_Model':
        model = model_class( input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, gcn_layers=gcn_layers, gcn_hidden_size=gcn_hidden_size)
    elif model_name == 'ConvTransformerModel':
        model = model_class(input_size=input_size, hidden_size=hidden_size, num_spatial_features=len(static_features), num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, kernel_size=kernel_size, stride=stride, n_heads=n_heads, dim_feedforward=dim_feedforward)
    elif model_name == 'ConvTFTModel':
            model = model_class(input_size=input_size, hidden_size=hidden_size, num_spatial_features=len(static_features), num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, kernel_size=kernel_size, stride=stride, n_heads=n_heads, dim_feedforward=dim_feedforward)
    elif model_name in ['DoubleConvTFTModel','DoubleConvTransformerModel']:
            model = model_class(input_size=input_size, hidden_size=hidden_size, num_spatial_features_group1=len(curv_columns), num_spatial_features_group2=len(slope_columns), num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, kernel_size=kernel_size, stride=stride, n_heads=n_heads, dim_feedforward=dim_feedforward)
    else:
        model = model_class(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay)

    # The rest of your function remains the same
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "checkpoints"),
        filename=f'best_model={model_name}-time={model_time}',
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=pat_train,
        mode="min",
        verbose=False, 
        min_delta=min_del_train
    )

    print('>>> TRAINING THE BEST MODEL:')
    
    trial_hyperparam_message_best_model= (  f"BEST MODEL | "
                                            f"Best trial {study.best_trial} | "
                                            f"Hyperparameters: hidden_size={hidden_size}, "
                                            f"num_layers={num_layers}, dropout_prob={dropout_prob:.6f}, "
                                            f"learning_rate={learning_rate:.8f}, weight_decay={weight_decay:.8f}, "
                                            f"n_heads={n_heads:.0f}, dim_feedforward={dim_feedforward:.0f}"
                                            f"gcn_layers={gcn_layers}, gcn_hidden_size={gcn_hidden_size}, kernel_size={kernel_size}, stride={kernel_size}")
                            
    print(trial_hyperparam_message_best_model)
    append_to_log(trial_hyperparam_message_best_model)
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        callbacks=[checkpoint_callback, early_stopping_callback, ClearMemoryCallback()], #,SimpleLogger()],
        enable_progress_bar=False,
        log_every_n_steps=5000,
        enable_model_summary=False,
        profiler=False,
        precision=32
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    try:
        # Save the trained model's state_dict
        model_save_path = os.path.join(save_dir, f"Best_model={model_name}_time={model_time}_iter{iterat}_trained.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Trained model saved to: {model_save_path}")
    except:
        print(f'>>> Model unsuccessfully saved.')
    
    # Manually clear CUDA memory
    torch.cuda.empty_cache()  # Free GPU memory

    # Manually trigger Python's garbage collector to free unused memory
    gc.collect()

    # Stop the timer
    end_time = time.time()

    # Calculate the duration and print it
    duration = end_time - start_time
    print(f"Training duration: {duration:.2f} seconds")

    #########################################################################################################################
    #########################################################################################################################

    print('=============================================================================')
    print('=============================================================================')

    # Define constants and configurations outside of the loop
    output_file = f'predictions_model_{model_name}_granularity_{model_time}_projection_1hour_iter{iterat}.csv'
    checkpoint_dir = "d:/Users/afuentes/Documents/GERARD/preMARIA-AP7-lstm-trc-3/checkpoints"

    # Initialize output file
    with open(output_file, 'w') as f:
        f.write("")

    # Precompute datetime conversions and scaling outside the loop
    df_merge['dat'] = pd.to_datetime(df_merge['dat'])
    df_merge_sim.loc[:, 'dat'] = pd.to_datetime(df_merge_sim['dat'])
    features_sim = df_merge_sim.drop(columns=cols_targets).drop(columns='dat')
    targets_sim = df_merge_sim[cols_targets]
    features_sim_scaled = scaler_features.transform(features_sim)

    if targets_sim.ndim == 1:
        targets_sim = targets_sim.values.reshape(-1, 1)  # Reshape to 2D if needed
    targets_sim_scaled = scaler_targets.transform(targets_sim)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        callbacks=[checkpoint_callback, early_stopping_callback, ClearMemoryCallback()], # Remove SimpleLogger()
        enable_progress_bar=False,  # Disable progress bar
        enable_model_summary=False,
        profiler=False,
        precision=32
    )

    # Main loop using the function
    start_time_sim = time.time()
    total_iterations = int((end_of_data_time - initial_train_end_time) / timedelta(hours=1))
    datetime_array = pd.date_range(start=initial_train_end_time, end=end_of_data_time, freq='W')

    for iteration, fine_tune_time in enumerate(datetime_array[:-1]):
        run_iteration(iteration, fine_tune_time, datetime_array, start_time_sim, total_iterations)


    # Function to safely delete variables
    def safe_del(var_name):
        if var_name in globals():
            del globals()[var_name]
        elif var_name in locals():
            del locals()[var_name]

    # Remove variables
    for var in [
        "hidden_size", "num_layers", "dropout_prob", "learning_rate", "weight_decay",
        "n_heads", "dim_feedforward", "model_class", "model", "HYPERPARAMS", "trainer", 
        "log_file", "study", "checkpoint_callback", "early_stopping_callback", "model_save_path",
        "start_time", "end_time", "duration", "output_file", "checkpoint_dir"]:
        safe_del(var)

    # Remove DataFrames
    for df_var in ["features_sim_scaled", "targets_sim_scaled"]:
        safe_del(df_var)


    # Clear other objects
    torch.cuda.empty_cache()  # Clear CUDA memory if used
    gc.collect()  # Trigger garbage collection to clean up memory
