import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import pytorch_lightning as pl

class DoubleConvLSTM(pl.LightningModule):
    def __init__(self, input_size, num_spatial_features_group1, num_spatial_features_group2,
                 hidden_size, num_layers, output_size, dropout_prob, learning_rate,
                 weight_decay, kernel_size, stride):
        super(DoubleConvLSTM, self).__init__()
        
        self.input_size = input_size
        self.num_spatial_features_group1 = num_spatial_features_group1
        self.num_spatial_features_group2 = num_spatial_features_group2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Convolutional layers for extracting spatial features
        self.conv1d_group1 = nn.Conv1d(in_channels=num_spatial_features_group1, 
                                       out_channels=hidden_size // 2,
                                       kernel_size=kernel_size, stride=stride, padding="same")
        
        self.conv1d_group2 = nn.Conv1d(in_channels=num_spatial_features_group2, 
                                       out_channels=hidden_size // 2,
                                       kernel_size=kernel_size, stride=stride, padding="same")
        
        # Linear layer for embedding temporal features
        self.temporal_embedding = nn.Linear(input_size - (num_spatial_features_group1 + num_spatial_features_group2), hidden_size)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        batch_size, time_steps, num_features = x.shape

        # Separate temporal features and spatial features
        temporal_features = x[:, :, :-(self.num_spatial_features_group1 + self.num_spatial_features_group2)]
        spatial_features_group1 = x[:, 0, -self.num_spatial_features_group1 - self.num_spatial_features_group2:-self.num_spatial_features_group2]
        spatial_features_group2 = x[:, 0, -self.num_spatial_features_group2:]

        # Convolutional feature extraction
        spatial_features_group1 = self.conv1d_group1(spatial_features_group1.unsqueeze(2)).squeeze(2)
        spatial_features_group2 = self.conv1d_group2(spatial_features_group2.unsqueeze(2)).squeeze(2)

        # Concatenate spatial features
        spatial_features = torch.cat([spatial_features_group1, spatial_features_group2], dim=-1)

        # Embed temporal features
        temporal_embedding = self.temporal_embedding(temporal_features)

        # Combine temporal and spatial features
        combined_features = temporal_embedding + spatial_features.unsqueeze(1)

        # LSTM processing
        lstm_out, _ = self.lstm(combined_features)
        final_output = lstm_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = self.fc(final_output)
        output = F.softplus(output)  # Ensure strictly positive output
        return output.unsqueeze(1)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        val_mae = torch.mean(torch.abs(outputs - labels))
        val_rmse = torch.sqrt(loss)
        self.log('val_loss', loss)
        self.log('val_mae', val_mae)
        self.log('val_rmse', val_rmse)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
