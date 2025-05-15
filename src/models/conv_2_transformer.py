import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim

class DoubleConvTransformerModel(pl.LightningModule):
    def __init__(self, 
                 input_size, 
                 num_spatial_features_group1, 
                 num_spatial_features_group2,
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
        super(DoubleConvTransformerModel, self).__init__()

        self.input_size = input_size
        self.num_spatial_features_group1 = num_spatial_features_group1
        self.num_spatial_features_group2 = num_spatial_features_group2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Calculate padding only if stride == 1
        padding1 = (kernel_size - 1) // 2 if stride == 1 else 0
        padding2 = (kernel_size - 1) // 2 if stride == 1 else 0

        # Convolutional layers for two spatial feature groups
        self.conv1d_group1 = nn.Conv1d(
            in_channels=num_spatial_features_group1,  
            out_channels=hidden_size // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding1
        )

        self.conv1d_group2 = nn.Conv1d(
            in_channels=num_spatial_features_group2,  
            out_channels=hidden_size // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding2
        )

        # Linear embedding for temporal features
        self.temporal_embedding = nn.Linear(
            input_size - (num_spatial_features_group1 + num_spatial_features_group2), 
            hidden_size
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        batch_size, time_steps, _ = x.shape

        # Split temporal and static features
        temporal_features = x[:, :, :-(self.num_spatial_features_group1 + self.num_spatial_features_group2)]
        group1 = x[:, 0, -self.num_spatial_features_group1 - self.num_spatial_features_group2:-self.num_spatial_features_group2]
        group2 = x[:, 0, -self.num_spatial_features_group2:]

        group1 = self.conv1d_group1(group1.unsqueeze(2)).squeeze(2)
        group2 = self.conv1d_group2(group2.unsqueeze(2)).squeeze(2)

        spatial_features = torch.cat([group1, group2], dim=-1)
        temporal_embedding = self.temporal_embedding(temporal_features)
        combined = temporal_embedding + spatial_features.unsqueeze(1)

        transformer_out = self.transformer_encoder(combined)
        final_output = transformer_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = self.fc(final_output)
        output = F.softplus(output)
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
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
