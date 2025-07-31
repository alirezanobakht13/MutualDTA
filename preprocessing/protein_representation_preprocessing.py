import torch
import json
import os
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass

import tyro
@dataclass
class Args:
    esm2_model: str = 'esm2_t33_650M_UR50D'
    "you can use larger models such as `esm2_t48_15B_UR50D` for better performance, but it requires more memory"
    dataset: str = 'davis'
    "dataset to process, can be 'davis' or 'kiba'"

hp = tyro.cli(Args)
    

# Load model and alphabet from pre-trained esm2 model by Facebook AI Research
model, alphabet = torch.hub.load("facebookresearch/esm:main", hp.esm2_model)

# Set the device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()  # Set model to evaluation mode (disables dropout for deterministic results)

# Load protein data
dataset = hp.dataset  # 'davis' or 'kiba'
data_path = f'data/{dataset}'
proteins = json.load(open(os.path.join(data_path, "proteins.txt")), object_pairs_hook=OrderedDict)  
proteins = [(name, seq) for name, seq in proteins.items()]

# Create a DataLoader for batched protein processing
dataloader = DataLoader(proteins, batch_size=2, shuffle=False, collate_fn=lambda x: x)

# Define the directory path to save processed protein data
save_protein_path = os.path.join(data_path, 'proteins')
os.makedirs(save_protein_path, exist_ok=True)

# Process each batch in the dataloader and save protein representations
for i, data in enumerate(dataloader):
    print(f"Processing batch {i+1} of {len(dataloader)}")
    save_path = os.path.join(save_protein_path, f"saved_{i}.pt")
    
    # Convert sequences to tokens
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=1024)
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # Generate embeddings with the model
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    token_representations = results["representations"][33]
    
    # Save the processed tokens
    torch.save(token_representations, save_path)

# Load saved protein data, process, and save in a single tensor file
proteins = []

for filename in os.listdir(save_protein_path):
    # Load individual protein tensor
    protein = torch.load(os.path.join(save_protein_path, filename))
    print(f"Loaded {filename}, original shape: {protein.shape}")
    
    # Remove start/end tokens and pad to a fixed size of 1024 tokens
    protein = protein[:, 1:protein.size(1) - 1, :]
    protein = F.pad(protein, [0, 0, 0, 1024 - protein.size(1)])
    
    proteins.append(protein)

# Concatenate all protein tensors and save as a single tensor file
proteins = torch.cat(proteins, dim=0)
print(f"Final concatenated protein shape: {proteins.shape}")
os.makedirs(os.path.join(data_path,'processed'),exist_ok=True)
torch.save(proteins, os.path.join(data_path,'processed','proteins.pt'))

print(f'You can now delete the folder: {save_protein_path}')