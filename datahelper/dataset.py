import os
import torch
import numpy as np
from torch.utils.data import Dataset

class DTADataset(Dataset):
    def __init__(self, root, dataset, mode, cold_start, device):
        assert mode in ['train','test']
        self.mode = mode
        self.device = torch.device(device)
        assert cold_start in ['','drug','protein','all','nfold5']
        if cold_start=='nfold5':
            cold_start = cold_start+'_'
        else:
            cold_start = cold_start+'_cold_start_' if cold_start else ''
        if mode =='train':
            data = torch.tensor(np.loadtxt(os.path.join(root, dataset,'processed','%strain.txt'%cold_start)))
        else:
            data = torch.tensor(np.loadtxt(os.path.join(root, dataset,'processed','%stest.txt'%cold_start)))
        data = data.to(device)
        self.compound_idx,self.protein_idx,self.Y = data[:,0].long().to(device),data[:,1].long().to(device),data[:,2].to(device)
        # first col is compound idx, the second is protein idx, the third col is the affinty (Y)
        
        self.compounds = torch.load(os.path.join(root, dataset,'processed','unimol_compounds.pt'))
        self.compounds = [compound.to(device) for compound in self.compounds]
        self.proteins = torch.load(os.path.join(root, dataset,'processed','encoded_proteins.pt'),device)
        self.proteins.requires_grad = False

    def __len__(self):
        return len(self.compound_idx)

    def __getitem__(self, idx):
        compound = self.compounds[self.compound_idx[idx]]
        protein = self.proteins[self.protein_idx[idx]]
        Y = self.Y[idx]
        return compound, protein, Y
    
    def collate_fn(self,batch):   

        compounds = [item[0] for item in batch]
        proteins = torch.stack([item[1] for item in batch])
        Y = torch.stack([item[2] for item in batch])
        
        return compounds, proteins, Y
    