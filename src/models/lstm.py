import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl


class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, learning_rate, weight_decay):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Define the LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Loss function
        self.criterion = nn.MSELoss() # Old
        # self.criterion = nn.HuberLoss(delta=delta_huber_loss) # New

    def forward(self, x):
        # Initialize the hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the output at the final timestep
        final_output = lstm_out[:, -1, :]
        
        # Apply dropout
        final_output = self.dropout(final_output)
        
        # Pass through the fully connected layer and apply ReLU activation
        output = self.fc(final_output)
        output = nn.ReLU()(output)  # Ensure positive outputs
        
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.view(-1, self.output_size)
        
        # Forward pass
        outputs = self(inputs)
        
        # Compute the loss
        loss = self.criterion(outputs, labels)
        
        # Log training loss
        self.log('train_loss', loss)
        return loss
    
    def on_train_epoch_end(self):
        # The training loss for the current epoch is already logged, so we just retrieve it and print
        epoch_loss = self.trainer.callback_metrics["train_loss"]
        # print(f"Epoch {self.current_epoch} - Loss: {epoch_loss:.4f}")

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.view(-1, self.output_size)  # Reshape labels to match output if needed
        # Forward pass
        outputs = self(inputs)
        
        # Compute the loss
        loss = self.criterion(outputs, labels)
        
        # Compute metrics (MAE, R2, RMSE)
        val_mae = mean_absolute_error(labels.cpu().numpy(), outputs.cpu().detach().numpy())
        val_r2 = r2_score(labels.cpu().numpy(), outputs.cpu().detach().numpy())
        val_rmse = np.sqrt(mean_squared_error(labels.cpu().numpy(), outputs.cpu().detach().numpy()))
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_mae', val_mae)
        self.log('val_r2', val_r2)
        self.log('val_rmse', val_rmse)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
