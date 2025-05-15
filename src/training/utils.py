# Built-in Libraries
import os
import time
import gc
from datetime import timedelta

# Data Handling
import pandas as pd
import numpy as np

# PyTorch
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Optuna for hyperparameter optimization
import optuna

# W&B for experiment tracking
import wandb

# Scikit-learn (used for scaling; referenced as scaler_features, scaler_targets)
from sklearn.preprocessing import StandardScaler

from src.data_file_names import *

def benchmark_dataloader(num_workers, dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    start = time.time()
    for _ in range(5):  # Load 5 batches to estimate speed
        for batch in loader:
            pass
    end = time.time()
    return end - start

# Function to append logs to the file
def append_to_log(message, log_file):
    with open(log_file, "a") as f:
        f.write(message + "\n")

def get_hyperparams(model_name):
    hyperparams_dict = {
        'LSTMModel': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "dropout_prob_range": (0.1, 0.5),
            "learning_rate_range": (1e-5, 1e-2),
            "weight_decay_range": (1e-6, 1e-4),
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "kernel_size": [0],
            "stride": [0],
            "n_heads": [0],
            "dim_feedforward": [0],
        },
        'DoubleConvLSTM': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "dropout_prob_range": (0.1, 0.5),
            "learning_rate_range": (1e-5, 1e-2),
            "weight_decay_range": (1e-6, 1e-4),
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "kernel_size": [1, 3],
            "stride": [1], 
            "n_heads": [0],
            "dim_feedforward": [0],
        },
        'LSTMAttModel': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "dropout_prob_range": (0.2, 0.6),
            "learning_rate_range": (1e-5, 1e-3),
            "weight_decay_range": (1e-6, 1e-4),
            "n_heads": [2, 4, 8, 16],
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "kernel_size": [0],
            "stride": [0],
            "dim_feedforward": [0],
        },
        'xLSTMModel': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "dropout_prob_range": (0.1, 0.5),
            "learning_rate_range": (1e-5, 1e-2),
            "weight_decay_range": (1e-6, 1e-3),
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "kernel_size": [0],
            "stride": [0],
            "n_heads": [0],
            "dim_feedforward": [0],
        },
        'xLSTMAttModel': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "dropout_prob_range": (0.2, 0.6),
            "learning_rate_range": (1e-6, 1e-3),
            "weight_decay_range": (1e-6, 1e-3),
            "n_heads": [2, 4, 8, 16],
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "kernel_size": [0],
            "stride": [0],
            "n_heads": [0],
            "dim_feedforward": [0],
        },
        'TransformerModel': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "n_heads": [2, 4, 8, 16],
            "dropout_prob_range": (0.1, 0.4),
            "learning_rate_range": (1e-5, 1e-3),
            "weight_decay_range": (1e-6, 1e-4),
            "dim_feedforward": [2, 4, 8, 16],
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "kernel_size": [0],
            "stride": [0],
        },
        'TFTModel': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "n_heads": [2, 4, 8, 16],
            "dropout_prob_range": (0.1, 0.4),
            "learning_rate_range": (1e-5, 1e-3),
            "weight_decay_range": (1e-6, 1e-4),
            "dim_feedforward": [2, 4, 8, 16],
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "kernel_size": [0],
            "stride": [0],
        },
        'DoubleConvTransformerModel': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "n_heads": [2, 4, 8, 16],
            "dropout_prob_range": (0.1, 0.4),
            "learning_rate_range": (1e-5, 1e-3),
            "weight_decay_range": (1e-6, 1e-4),
            "dim_feedforward": [2, 4, 8, 16],
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "kernel_size": [1, 3],
            "stride": [1], 
        },
        'Conv_LSTM_Model': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "dropout_prob_range": (0.1, 0.5),  # Dropout probabilities
            "learning_rate_range": (1e-5, 1e-3),  # Learning rate for Adam optimizer
            "weight_decay_range": (1e-6, 1e-4),  # L2 regularization (weight decay)
            "kernel_size": [1, 3],  # Size of the CNN kernel
            "stride": [1],  # Stride size for the Conv1D layer
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "n_heads": [0],
            "dim_feedforward": [0],
        },
        'GCN_LSTM_Model': {
            "hidden_size": [128, 256, 512],
            "num_layers": [1, 2, 3, 4],  # Number of LSTM layers
            "gcn_layers": [1, 2, 3],  # Number of GCN layers
            "dropout_prob_range": (0.1, 0.5),  # Dropout probabilities
            "learning_rate_range": (1e-5, 1e-3),  # Learning rate for Adam optimizer
            "weight_decay_range": (1e-6, 1e-4),  # L2 regularization (weight decay)
            "gcn_hidden_size": [32, 64, 128, 256],  # Hidden size of GCN layers
            "kernel_size": [0],
            "stride": [0],
            "n_heads": [0],
            "dim_feedforward": [0],
        },
        'ConvTransformerModel': {
            "hidden_size": [128, 256, 512],  # LSTM hidden size
            "num_layers": [3, 4, 5],  # Number of LSTM layers
            "dropout_prob_range": (0.1, 0.5),  # Dropout probabilities
            "learning_rate_range": (1e-5, 1e-3),  # Learning rate for Adam optimizer
            "weight_decay_range": (1e-6, 1e-4),  # L2 regularization (weight decay)
            "kernel_size": [1, 3],  # Size of the CNN kernel
            "stride": [1],  # Stride size for the Conv1D layer
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "n_heads": [2, 4, 8, 16],
            "dim_feedforward": [2, 4, 8, 16],
        },
        'ConvTFTModel': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "dropout_prob_range": (0.1, 0.5),  # Dropout probabilities
            "learning_rate_range": (1e-5, 1e-3),  # Learning rate for Adam optimizer
            "weight_decay_range": (1e-6, 1e-4),  # L2 regularization (weight decay)
            "kernel_size": [1, 3],  # Size of the CNN kernel
            "stride": [1],  # Stride size for the Conv1D layer
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "n_heads": [2, 4, 8, 16],
            "dim_feedforward": [2, 4, 8, 16],
        },
        'DoubleConvTFTModel': {
            "hidden_size": [128, 256, 512],
            "num_layers": [3, 4, 5],
            "dropout_prob_range": (0.1, 0.5),  # Dropout probabilities
            "learning_rate_range": (1e-5, 1e-3),  # Learning rate for Adam optimizer
            "weight_decay_range": (1e-6, 1e-4),  # L2 regularization (weight decay)
            "kernel_size": [1, 3],  # Size of the CNN kernel
            "stride": [1],  # Stride size for the Conv1D layer
            "gcn_layers": [0],
            "gcn_hidden_size": [0],
            "n_heads": [2, 4, 8, 16],
            "dim_feedforward": [2, 4, 8, 16],
        }
    }
    return hyperparams_dict.get(model_name, {
        "hidden_size": [128, 256, 512],
        "num_layers": [3, 4, 5, 6],
        "dropout_prob_range": (0.2, 0.6),
        "learning_rate_range": (1e-6, 1e-3),
        "weight_decay_range": (1e-6, 1e-3),
    })


def create_objective(
    model_name,
    model_mapping,
    HYPERPARAMS,
    WANDB_PROJECT,
    WANDB_ENTITY,
    input_size,
    output_size,
    train_loader,
    val_loader,
    static_features,
    curv_columns,
    slope_columns,
    log_file,
    iterat,
    model_time,
    pat_train,
    min_del_train,
    epochs
        ):
    
    def objective(trial, HYPERPARAMS):
        try:
            append_to_log("==============================================")
            append_to_log(f"Starting trial {trial.number}")

            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"{model_name}_trial_{trial.number}",
                dir="./wandb_logs",  # Logs will be stored here
                config={
                    "model_name_iter": model_name+"_"+iterat, 
                    "hidden_size": trial.suggest_categorical("hidden_size", HYPERPARAMS["hidden_size"]),
                    "num_layers": trial.suggest_categorical("num_layers", HYPERPARAMS["num_layers"]),
                    "dropout_prob": trial.suggest_float("dropout_prob", *HYPERPARAMS["dropout_prob_range"], log=True),
                    "learning_rate": trial.suggest_float("learning_rate", *HYPERPARAMS["learning_rate_range"], log=True),
                    "weight_decay": trial.suggest_float("weight_decay", *HYPERPARAMS["weight_decay_range"], log=True),
                    "gcn_layers": trial.suggest_categorical("gcn_layers", HYPERPARAMS["gcn_layers"]),
                    "gcn_hidden_size": trial.suggest_categorical("gcn_hidden_size", HYPERPARAMS["gcn_hidden_size"]),
                    "kernel_size": trial.suggest_categorical("kernel_size", HYPERPARAMS["kernel_size"]),
                    "stride": trial.suggest_categorical("stride", HYPERPARAMS["stride"]),
                    "n_heads": trial.suggest_categorical("n_heads", HYPERPARAMS["n_heads"]),
                    "dim_feedforward": trial.suggest_categorical("dim_feedforward", HYPERPARAMS["dim_feedforward"])
                },
                settings=wandb.Settings(silent=True)  # Suppresses logging output
            )
            
            # Define hyperparameters
            hidden_size = wandb.config.hidden_size
            num_layers = wandb.config.num_layers
            dropout_prob = wandb.config.dropout_prob
            learning_rate = wandb.config.learning_rate
            weight_decay = wandb.config.weight_decay
            gcn_layers = wandb.config.gcn_layers
            gcn_hidden_size = wandb.config.gcn_hidden_size
            kernel_size = wandb.config.kernel_size
            stride = wandb.config.stride
            dim_feedforward = wandb.config.dim_feedforward
            n_heads = wandb.config.n_heads

            model_class = model_mapping[model_name]  # model_name should be set to one of the four model names

            if model_name in ['TransformerModel','TFTModel']:
                model = model_class(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, n_heads=n_heads, dim_feedforward=dim_feedforward)
            elif model_name in ['LSTMAttModel', 'xLSTMAttModel']:
                model = model_class(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, n_heads=n_heads)
            elif model_name == 'DoubleConvLSTM':
                model = model_class(input_size=input_size, hidden_size=hidden_size, num_spatial_features_group1=len(curv_columns), num_spatial_features_group2=len(slope_columns), num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, kernel_size=kernel_size, stride=stride)
            elif model_name == 'Conv_LSTM_Model':
                model = model_class(input_size=input_size, hidden_size=hidden_size, num_spatial_features=len(static_features), num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, kernel_size=kernel_size, stride=stride)
            elif model_name == 'GCN_LSTM_Model':
                model = model_class(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, learning_rate=learning_rate, weight_decay=weight_decay, gcn_layers=gcn_layers, gcn_hidden_size=gcn_hidden_size)
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
                filename=f'best_model={model_name}-time={model_time}_iter{iterat}',
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


            # Function to configure optimizers with learning rate scheduling
            def configure_optimizers(model, learning_rate, weight_decay):
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                # Reduce LR when validation loss stops improving
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6, verbose=True
                )

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",  # Reduce LR based on validation loss
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            from pytorch_lightning.loggers import CSVLogger
            logger = CSVLogger("logs", name="my_model")
            # Modify the training process to use the scheduler
            trainer = pl.Trainer(
                max_epochs=epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                logger=logger,
                callbacks=[
                    ModelCheckpoint(dirpath="checkpoints", monitor="val_loss", mode="min", save_top_k=1),
                    EarlyStopping(monitor="val_loss", patience=pat_train, mode="min", verbose=False, min_delta=min_del_train),
                    LearningRateMonitor(logging_interval='step',),  # Monitor LR changes
                    ClearMemoryCallback(),
                ],
                enable_progress_bar=False,
                log_every_n_steps=5000,
                enable_model_summary=False,
                profiler=False,
                precision=32
            )

            trainer.fit(model, train_loader, val_loader)

            val_loss = trainer.callback_metrics['val_loss'].item()
            train_loss = trainer.callback_metrics['train_loss'].item() if 'train_loss' in trainer.callback_metrics else None

            # Log metrics to W&B
            wandb.log({
                "val_loss": val_loss,
                "train_loss": train_loss,
                "epochs": trainer.current_epoch,
            })
            
            trial_message = (
                f"Trial {trial.number} completed with val_loss: {val_loss:.6f} in {trainer.current_epoch + 1} epochs | "
                f"Hyperparameters: hidden_size={hidden_size}, num_layers={num_layers}, "
                f"dropout_prob={dropout_prob:.6f}, learning_rate={learning_rate:.8f}, weight_decay={weight_decay:.8f}, "
                f"n_heads={n_heads:.0f}, dim_feedforward={dim_feedforward:.0f}, "
                f"gcn_layers={gcn_layers}, gcn_hidden_size={gcn_hidden_size}, kernel_size={kernel_size}, stride={stride}"
            )
            # Track additional metrics for analysis
            train_time = trainer.current_epoch  # Number of epochs taken

            # Log additional metrics if applicable (e.g., generalization gap)
            generalization_gap = abs(train_loss - val_loss) if train_loss is not None else None
            if generalization_gap is not None:
                wandb.log({"generalization_gap": generalization_gap})

            # Finish W&B run
            wandb.finish()

            trial_message += f" | Additional Metrics: train_time={train_time}, generalization_gap={generalization_gap}"

            print(trial_message)
            append_to_log(trial_message)
            
            torch.cuda.empty_cache()
            del model, trainer
            gc.collect()

            return val_loss
        
        except RuntimeError as e:
            error_message = f"Trial {trial.number} failed due to {e}"
            print(f'>>>> ERROR: {error_message}')
            append_to_log(error_message)
            return float('inf')  # Penalize failing trials


# Function to estimate time remaining
def estimate_time_remaining(iteration, total_iterations, start_time_sim):
    elapsed_time = time.time() - start_time_sim
    remaining_iterations = total_iterations - (iteration + 1)
    avg_time_per_iteration = elapsed_time / (iteration + 1)
    remaining_time = remaining_iterations * avg_time_per_iteration
    return remaining_time


class ClearMemoryCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        print(f"Clearing cache and collecting garbage after epoch {trainer.current_epoch}")
        torch.cuda.empty_cache()  # Clear GPU cache
        gc.collect()  # Collect garbage


# Core function for each iteration
def run_iteration(
    iteration, fine_tune_time, datetime_array, start_time_sim, total_iterations,
    df_merge_sim, features_sim_scaled, targets_sim_scaled,
    train_size, data_load_batch, datload_num_work,
    trainer, model, df_merge, scaler_features, scaler_targets,
    cols_targets, output_file, estimate_time_remaining
):

    # Define time range mask
    current_time = fine_tune_time + timedelta(weeks=1)
    mask = (df_merge_sim['dat'] >= fine_tune_time) & (df_merge_sim['dat'] < current_time)

    # Extract relevant hourly data if available
    features_hourly = features_sim_scaled[mask]
    targets_hourly = targets_sim_scaled[mask]

    if features_hourly.size > 0:
        # Prepare data tensors
        features_hourly_tensor = torch.tensor(features_hourly, dtype=torch.float32).unsqueeze(1)
        targets_hourly_tensor = torch.tensor(targets_hourly, dtype=torch.float32).unsqueeze(1)
        num_sim = int(len(features_hourly_tensor) * train_size)

        # Use persistent data loaders
        train_hourly_loader = DataLoader(
            TensorDataset(features_hourly_tensor[:num_sim], targets_hourly_tensor[:num_sim]),
            batch_size=data_load_batch, shuffle=False, num_workers=datload_num_work, persistent_workers=True
        )
        val_hourly_loader = DataLoader(
            TensorDataset(features_hourly_tensor[num_sim:], targets_hourly_tensor[num_sim:]),
            batch_size=data_load_batch, shuffle=False, num_workers=datload_num_work, persistent_workers=True
        )

        # Train the model
        trainer.fit(model, train_hourly_loader, val_hourly_loader)
    
        # Generate future predictions
        future_times = pd.date_range(start=fine_tune_time, end=current_time, freq='1h')
        pred_feat = df_merge[df_merge['dat'].isin(future_times)]
        if not pred_feat.empty:
            current_features_normalized = scaler_features.transform(pred_feat.drop(columns=cols_targets).drop(columns='dat'))
            current_features_tensor = torch.tensor(current_features_normalized, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                prediction = model(current_features_tensor)
                prediction_np = prediction.squeeze().cpu().numpy()
                if prediction_np.ndim == 1:
                    prediction_np = prediction_np.reshape(-1, 1)  # Ensure 2D shape
                prediction_rescaled = scaler_targets.inverse_transform(prediction_np)
            # Save predictions
            df_prediction_predtarg = pd.DataFrame(data=prediction_rescaled, columns=[f"{name}_pred" for name in cols_targets])
            df_prediction_predtarg['Fine-tune Time'] = fine_tune_time
            df_prediction = pd.concat([pred_feat.reset_index(drop=True), df_prediction_predtarg.reset_index(drop=True)], axis=1)
            df_prediction.to_csv(output_file, mode='a', sep=";", decimal=".", header=iteration == 0, encoding="latin-1", index=False)

    # Periodically log progress
    if iteration % 10 == 0:
        remaining_time = estimate_time_remaining(iteration, total_iterations, start_time_sim)
        print(f"=== Iteration {iteration}/{total_iterations} ===")
        print(f">>> Estimated time remaining: {timedelta(seconds=int(remaining_time))}")
        print("===========================================")
