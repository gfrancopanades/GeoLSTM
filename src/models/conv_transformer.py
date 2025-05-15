import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

class ConvTransformerModel(pl.LightningModule):
    def __init__(self, 
                 input_size, 
                 num_spatial_features, 
                 hidden_size, 
                 num_layers, 
                 output_size, 
                 dropout_prob, 
                 learning_rate, 
                 weight_decay, 
                 kernel_size, 
                 stride, 
                 n_heads, 
                 dim_feedforward):
        super(ConvTransformerModel, self).__init__()
        
        self.input_size = input_size
        self.num_spatial_features = num_spatial_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Convolutional layer for spatial features
        self.conv1d = nn.Conv1d(
            in_channels=num_spatial_features,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding="same"
        )

        # Embedding layer for temporal features
        self.input_embedding = nn.Linear(input_size - num_spatial_features + hidden_size, hidden_size)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Loss function
        self.criterion = nn.MSELoss()

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
        spatial_features = torch.relu(spatial_features)

        # Transpose back: (batch_size, time_steps, hidden_size)
        spatial_features = spatial_features.permute(0, 2, 1)

        # Combine temporal and spatial features
        combined_features = torch.cat([temporal_features, spatial_features], dim=-1)

        # Project combined features to hidden size
        embedded_features = self.input_embedding(combined_features)

        # Pass through Transformer
        transformer_out = self.transformer_encoder(embedded_features)

        # Take the output at the final timestep
        final_output = transformer_out[:, -1, :]

        # Apply dropout and final linear layer
        final_output = self.dropout(final_output)
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
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
