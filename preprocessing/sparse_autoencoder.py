import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

import tyro
from dataclasses import dataclass

@dataclass
class Config:
    dataset_name: str = 'davis'    # Dataset name
    input_dim: int = 1280          # Input dimension
    hidden_dim: int = 256          # Hidden layer dimension
    lr: float = 0.001              # Learning rate
    alpha: float = 1e-8            # L1 regularization parameter
    num_epochs: int = 80           # Number of training epochs
    batch_size: int = 64           # Batch size
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device

config = tyro.cli(Config)
dataset_name = config.dataset_name
input_dim = config.input_dim
hidden_dim = config.hidden_dim
lr = config.lr
alpha = config.alpha
num_epochs = config.num_epochs
batch_size = config.batch_size
device = config.device

# Define a simple MLP-based autoencoder model
class SimpleMLPAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha=1e-8):
        super(SimpleMLPAutoEncoder, self).__init__()
        
        # Encoder: Maps input dimension to hidden dimension
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Decoder: Restores hidden dimension back to input dimension
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.alpha = alpha

    def forward(self, x):
        # Encoding stage
        encoded = self.encode(x)
        
        # Decoding stage (optionally, activation functions can be added here)
        decoded = self.decode(encoded)
        
        return decoded
    
    def decode(self, x):
        return self.decoder(x)
    
    def encode(self, x):
        return self.encoder(x)

# Dataset class for loading protein data
class ProteinDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.data = torch.load(root, map_location=device)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

# Initialize model, dataset, and dataloader
model = SimpleMLPAutoEncoder(input_dim, hidden_dim, alpha).to(device)
dataset = ProteinDataset(f'data/{dataset_name}/processed/proteins.pt')
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the optimizer (e.g., Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in trange(num_epochs, desc="Training"):
    try:
        reconstruction_metric = []
        l1_metric = []
        for data in train_dataloader:
            # Load batch data to the device
            inputs = data.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss (mean squared error + L1 regularization)
            reconstruction_loss = nn.MSELoss()(outputs, inputs)
            l1_loss = alpha * (torch.abs(model.encoder.weight).sum() + torch.abs(model.decoder.weight).sum())
            loss = reconstruction_loss + l1_loss
            reconstruction_metric.append(reconstruction_loss.item())
            l1_metric.append(l1_loss.item())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print average losses for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}] - Reconstruction Loss: {sum(reconstruction_metric)/len(reconstruction_metric):.4f} | L1 Loss: {sum(l1_metric)/len(l1_metric):.4f}')
    
    except KeyboardInterrupt:
        print("Training interrupted manually.")
        break

# Encoding step
print("Start encoding...")
encoded = []
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for data in dataloader:
        # Load batch data to the device
        inputs = data.to(device)
        
        # Encode inputs and store the results
        outputs = model.encode(inputs).cpu()
        encoded.append(outputs)

# Concatenate and save encoded data
encoded = torch.cat(encoded)
torch.save(encoded, f'data/{dataset_name}/processed/encoded_proteins.pt')        
print("Encoding complete!")
