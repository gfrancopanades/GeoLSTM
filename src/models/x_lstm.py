import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl


class xLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(xLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM Gates
        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        
        # Exponential gating
        self.exp_gating = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, x, hidden):
        h, c = hidden
        gates = self.i2h(x) + self.h2h(h)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        
        # Apply exponential gating to control memory cell updates
        input_gate = torch.sigmoid(input_gate) * torch.exp(self.exp_gating[:self.hidden_size])
        forget_gate = torch.sigmoid(forget_gate) * torch.exp(self.exp_gating[self.hidden_size:2*self.hidden_size])
        cell_gate = torch.tanh(cell_gate) * torch.exp(self.exp_gating[2*self.hidden_size:3*self.hidden_size])
        output_gate = torch.sigmoid(output_gate) * torch.exp(self.exp_gating[3*self.hidden_size:])
        
        # Update cell state and hidden state
        c_next = forget_gate * c + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)
        
        return h_next, c_next

class xLSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, learning_rate, weight_decay):
        super(xLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Define xLSTM layers
        self.xlstm_layers = nn.ModuleList([xLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Loss function
        self.criterion = nn.MSELoss() # Old
        # self.criterion = nn.HuberLoss(delta=delta_huber_loss) # New

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states for each layer
        h, c = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)], [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        
        # xLSTM forward pass through each time step
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.xlstm_layers[layer](x_t, (h[layer], c[layer]))
                x_t = h[layer]
            outputs.append(x_t.unsqueeze(1))
        
        # Concatenate outputs across time steps
        outputs = torch.cat(outputs, dim=1)
        
        # Take the output at the final timestep
        final_output = outputs[:, -1, :]
        
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
