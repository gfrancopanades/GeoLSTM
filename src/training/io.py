import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


import time
from joblib import dump, load

from data_file_names import *

def check_and_create_folder(path):
    # Check if the path exists
    if not os.path.exists(path):
        # Create the folder if it doesn't exist
        os.makedirs(path)
        print(f"Folder created at: {path}")
    else:
        print(f"Folder already exists at: {path}")


def feature_normalization(features):

    # Normalize features
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    file_path_scaler = os.path.join(folder_model, f'scaler_{model_version}.csv')
    print(file_path_scaler)
    # Save the scaler
    dump(scaler, file_path_scaler)

    return features_normalized

def splitting_data(combined_data):

    # Split the data into training, validation, and test sets
    train_size, val_size = 0.5, 0.2  # 50% training, 20% validation, and 30% for testing
    n = combined_data.shape[0]

    train = combined_data[:int(n * train_size)]
    val = combined_data[int(n * train_size):int(n * (train_size + val_size))]
    test = combined_data[int(n * (train_size + val_size)):]

    return train, val, test



def create_dataset_structure_for_LSTM(target_columns,data, input_sequence_length):
    X, y = [], []
    for i in range(len(data) - input_sequence_length):
        X.append(data[i:(i + input_sequence_length), :-len(target_columns)])
        y.append(data[i + input_sequence_length, -len(target_columns):])
    return np.array(X), np.array(y)



def save_array(array, folder_model, file_name):
    # Ensure the folder exists
    os.makedirs(folder_model, exist_ok=True)

    # Save array shape
    shape_file = os.path.join(folder_model, f"{file_name}_shape.csv")
    np.savetxt(shape_file, np.array(array.shape, dtype=int), fmt='%i', delimiter=',')

    # Flatten and save the array if it's more than 2D
    data_file = os.path.join(folder_model, f"{file_name}.csv")
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    np.savetxt(data_file, array, delimiter=';', fmt='%.18e', encoding='latin-1')


def save_all_model_data(X_train,y_train,X_val,y_val,X_test,y_test,model_version):
    # Function to save a numpy array to a CSV file and its shape to a separate file
    # Save each array to a CSV file
    save_array(X_train, folder_model, 'X_train_'+model_version)
    save_array(y_train, folder_model, 'y_train_'+model_version)
    save_array(X_val, folder_model, 'X_val_'+model_version)
    save_array(y_val, folder_model, 'y_val_'+model_version)
    save_array(X_test, folder_model, 'X_test_'+model_version)
    save_array(y_test, folder_model, 'y_test_'+model_version)




def save_model_history(history):
    history_dict = history.history # Get the dictionary containing each metric and the loss for each epoch
    json.dump(history_dict, open(os.path.join(folder_model, f'history_model_{model_version}.csv'), 'w')) # Save it under the form of a json file
    return history_dict

def plot_target_histogram(df_column):

    # Plotting a histogram from the 'Values' column
    plt.hist(df_column, bins=10, color='blue', alpha=0.7)

    # Adding labels and title
    plt.xlabel('Target value')
    plt.ylabel('Frequency')
    plt.title('Histogram of target')

    # Displaying the plot
    plt.savefig(os.path.join(folder_visualizations,f'histogram_target_plot_{model_version}.png'))  # Save the plot as an image file
    plt.close()  # Close the figure to free up memory


def plot_model_loss(model_version,history_dict):

    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust width to 1.5 times the previous size

    # Plot training & validation loss values
    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])

    # Add annotations for train accuracy values
    for i, val in enumerate(history_dict['loss']):
        plt.annotate(f'{val:.2f}', (i, val), textcoords='offset points', xytext=(0, 10), ha='center')

    # Add annotations for test accuracy values
    for i, val in enumerate(history_dict['val_loss']):
        plt.annotate(f'{val:.2f}', (i, val), textcoords='offset points', xytext=(0, 10), ha='center')
        
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(folder_visualizations,f'loss_plot_{model_version}.png'))  # Save the plot as an image file
    plt.close()  # Close the figure to free up memory



def plot_model_MeanSquaredError(model_version,history_dict):

    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust width to 1.5 times the previous size

    # Plot training & validation accuracy values
    plt.plot(history_dict['mean_squared_error'])
    plt.plot(history_dict['val_mean_squared_error'])

    # Add annotations for train accuracy values
    for i, val in enumerate(history_dict['mean_squared_error']):
        plt.annotate(f'{val:.2f}', (i, val), textcoords='offset points', xytext=(0, 10), ha='center')

    # Add annotations for test accuracy values
    for i, val in enumerate(history_dict['val_mean_squared_error']):
        plt.annotate(f'{val:.2f}', (i, val), textcoords='offset points', xytext=(0, 10), ha='center')

    plt.title('Model Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(folder_visualizations,f'mse_plot_{model_version}.png'))  # Save the plot as an image file
    plt.close()  # Close the figure to free up memory

def plot_model_MeanAbsoluteError(model_version,history_dict):

    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust width to 1.5 times the previous size

    # Plot training & validation accuracy values
    plt.plot(history_dict['mean_absolute_error'])
    plt.plot(history_dict['val_mean_absolute_error'])

    # Add annotations for train accuracy values
    for i, val in enumerate(history_dict['mean_absolute_error']):
        plt.annotate(f'{val:.2f}', (i, val), textcoords='offset points', xytext=(0, 10), ha='center')

    # Add annotations for test accuracy values
    for i, val in enumerate(history_dict['val_mean_absolute_error']):
        plt.annotate(f'{val:.2f}', (i, val), textcoords='offset points', xytext=(0, 10), ha='center')

    plt.title('Model Mean Absolute Error')
    plt.ylabel('Mean Absolut eError')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(folder_visualizations,f'MAE_plot_{model_version}.png'))  # Save the plot as an image file
    plt.close()  # Close the figure to free up memory

def plot_model_RootMeanSquaredError(model_version,history_dict):

    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust width to 1.5 times the previous size

    # Plot training & validation accuracy values
    plt.plot(history_dict['root_mean_squared_error'])
    plt.plot(history_dict['val_root_mean_squared_error'])

    # Add annotations for train accuracy values
    for i, val in enumerate(history_dict['root_mean_squared_error']):
        plt.annotate(f'{val:.2f}', (i, val), textcoords='offset points', xytext=(0, 10), ha='center')

    # Add annotations for test accuracy values
    for i, val in enumerate(history_dict['val_root_mean_squared_error']):
        plt.annotate(f'{val:.2f}', (i, val), textcoords='offset points', xytext=(0, 10), ha='center')

    plt.title('Model Root Mean Squared Error')
    plt.ylabel('Root Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(folder_visualizations,f'rmse_plot_{model_version}.png'))  # Save the plot as an image file
    plt.close()  # Close the figure to free up memory


# def r_squared(y_true, y_pred):
#     SS_res =  K.sum(K.square(y_true - y_pred)) 
#     SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
#     return (1 - SS_res/(SS_tot + K.epsilon()))


def plot_model_RSquared(model_version,history_dict):
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust width to 1.5 times the previous size

    # Plot training & validation accuracy values
    plt.plot(history_dict['r_squared'])
    plt.plot(history_dict['val_r_squared'])

    # Add annotations for train accuracy values
    for i, val in enumerate(history_dict['r_squared']):
        plt.annotate(f'{val:.2f}', (i, val), textcoords='offset points', xytext=(0, 10), ha='center')

    # Add annotations for test accuracy values
    for i, val in enumerate(history_dict['val_r_squared']):
        plt.annotate(f'{val:.2f}', (i, val), textcoords='offset points', xytext=(0, 10), ha='center')

    plt.title('Model R-squared')
    plt.ylabel('R-squared')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(folder_visualizations,f'rsquared_plot_{model_version}.png'))  # Save the plot as an image file
    plt.close()  # Close the figure to free up memory


###############################################################
###################### PYTORCH FUNCTIONS ######################
###############################################################

def train_and_validate_model(model, device, train_loader, val_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'train_mse': [],
        'val_mse': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_r2': [],
        'val_r2': []
    }

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            update_history(history, 'train', outputs.detach(), targets.detach())

        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                update_history(history, 'val', outputs.detach(), targets.detach())

    return history

def update_history(history, prefix, predicted, actual):
    history[f'{prefix}_loss'].append(float(predicted.size(0)))  # Collect batch size for debugging
    predicted = predicted.cpu().numpy()
    actual = actual.cpu().numpy()
    history[f'{prefix}_mae'].append(mean_absolute_error(actual, predicted))
    history[f'{prefix}_mse'].append(mean_squared_error(actual, predicted))
    history[f'{prefix}_rmse'].append(np.sqrt(history[f'{prefix}_mse'][-1]))
    history[f'{prefix}_r2'].append(r2_score(actual, predicted))

    return history


# Define the plotting function
def plot_metrics(history, folder_visualizations, model_version):
    metrics = {
        'accuracy': 'Accuracy',
        'loss': 'Loss',
        'mae': 'Mean Absolute Error',
        'mse': 'Mean Squared Error',
        'rmse': 'Root Mean Squared Error',
        'r2': 'R-squared'
    }

    os.makedirs(folder_visualizations, exist_ok=True)

    for metric, title in metrics.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(history[f'train_{metric}'], label=f'Train {metric.upper()}')
        plt.plot(history[f'val_{metric}'], label=f'Val {metric.upper()}')

        # Add annotations for train values
        for idx, val in enumerate(history[f'train_{metric}']):
            plt.annotate(f'{val:.2f}', (idx, val), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Add annotations for validation values
        for idx, val in enumerate(history[f'val_{metric}']):
            plt.annotate(f'{val:.2f}', (idx, val), textcoords="offset points", xytext=(0,10), ha='center')

        plt.title(f'Model {title}')
        plt.ylabel(title)
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(folder_visualizations, f'{metric}_plot_{model_version}.png'))
        plt.close()  # Close the figure to free up memory




def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in test_loader:
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                 torch.zeros(1, 1, model.hidden_layer_size).to(device))
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs.float())
            loss = criterion(output, targets.float())
    print(f'Test Loss: {loss.item()}')

def create_tensors(data):
    inputs = torch.Tensor(data[:, :-1])
    targets = torch.Tensor(data[:, -1])
    return TensorDataset(inputs, targets)


### SAVING AND LOADING 

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_history(history, path):
    with open(path, 'w') as f:
        json.dump(history, f)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model

def load_history(path):
    with open(path, 'r') as f:
        history = json.load(f)
    return history

