import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

class ConvTFTModel(pl.LightningModule):
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
        super(ConvTFTModel, self).__init__()

        self.input_size = input_size
        self.num_spatial_features = num_spatial_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Convolutional layer for spatial features (static)
        self.conv1d = nn.Conv1d(
            in_channels=num_spatial_features,  # Spatial features only
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding="same"
        )

        # Linear layer for embedding temporal features
        self.temporal_embedding = nn.Linear(input_size - num_spatial_features, hidden_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Temporal Attention Layer
        self.temporal_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, batch_first=True)

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # x shape: (batch_size, time_steps, num_features)
        batch_size, time_steps, num_features = x.shape

        # Separate temporal features and spatial features (static)
        temporal_features = x[:, :, :-self.num_spatial_features]  # Temporal features
        spatial_features = x[:, 0, -self.num_spatial_features:]  # Spatial features (same across time steps)

        # Transpose spatial features for Conv1D: (batch_size, num_spatial_features, time_steps)
        spatial_features = spatial_features.unsqueeze(2)  # Add a time dimension
        spatial_features = self.conv1d(spatial_features)  # Shape: (batch_size, hidden_size, 1)
        spatial_features = spatial_features.squeeze(2)  # Remove the singleton time dimension

        # Embed temporal features
        temporal_embedding = self.temporal_embedding(temporal_features)

        # Combine temporal and spatial features
        combined_features = temporal_embedding + spatial_features.unsqueeze(1)

        # Transformer encoder
        transformer_out = self.transformer_encoder(combined_features)

        # Temporal attention
        attn_out, _ = self.temporal_attention(transformer_out, transformer_out, transformer_out)

        # Final output from the last time step
        final_output = attn_out[:, -1, :]

        # Apply dropout and fully connected layer
        final_output = self.dropout(final_output)
        output = self.fc(final_output)
        
        # Apply Softplus to ensure strictly positive output
        output = F.softplus(output)

        output = output.unsqueeze(1)  # Shape: (batch_size, 1, num_horizons)

        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        # Forward pass
        outputs = self(inputs)

        # Compute loss
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        # Forward pass
        outputs = self(inputs)

        # Compute loss
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
