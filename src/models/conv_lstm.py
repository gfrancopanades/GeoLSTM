import torch
import torch.nn as nn
import pytorch_lightning as pl

class ConvLSTMModel(pl.LightningModule):
    def __init__(self, input_size, num_spatial_features, hidden_size, num_layers, output_size, dropout_prob, learning_rate, weight_decay, kernel_size, stride):
        super(ConvLSTMModel, self).__init__()

        self.input_size = input_size
        self.num_spatial_features = num_spatial_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Convolutional layer applied to spatial features
        self.conv1d = nn.Conv1d(
            in_channels=num_spatial_features,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding="same"
        )

        # Calculate the number of features after concatenation
        self.lstm_input_size = input_size - num_spatial_features + hidden_size

        # LSTM layer for temporal features
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Loss function
        self.criterion = nn.MSELoss() # Old
        # self.criterion = nn.HuberLoss(delta=delta_huber_loss) # New

    def forward(self, x):
        # x shape: (batch_size, time_steps, num_features)
        batch_size, time_steps, num_features = x.shape

        # Separate temporal and spatial features
        temporal_features = x[:, :, :-self.num_spatial_features]
        spatial_features = x[:, :, -self.num_spatial_features:]

        # Transpose for Conv1D: (batch_size, num_spatial_features, time_steps)
        spatial_features = spatial_features.permute(0, 2, 1)

        # Apply convolution
        spatial_features = self.conv1d(spatial_features)

        # Apply activation function
        spatial_features = torch.relu(spatial_features)

        # Transpose back for LSTM: (batch_size, time_steps, hidden_size)
        spatial_features = spatial_features.permute(0, 2, 1)

        # Combine temporal and processed spatial features
        x = torch.cat([temporal_features, spatial_features], dim=-1)

        # Apply LSTM
        lstm_out, _ = self.lstm(x)

        # Take the output at the final timestep
        final_output = lstm_out[:, -1, :]

        # Apply dropout
        final_output = self.dropout(final_output)

        # Pass through the fully connected layer
        output = self.fc(final_output)

        return output


    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.view(-1, self.output_size)

        # Forward pass
        outputs = self(inputs)

        # Compute the loss
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.view(-1, self.output_size)

        # Forward pass
        outputs = self(inputs)

        # Compute the loss and other metrics
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
