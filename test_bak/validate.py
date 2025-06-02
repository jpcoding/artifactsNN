import sys  
import os 
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))) 
from data_loader import BinaryNumpyDataset, PairedBinaryNumpyDataset
from torch_arcnn import ARCNN 
from torch.utils.data import DataLoader,random_split
import torch 
from torch import nn, optim 
import glob
import argparse



model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
print(f"Test Loss: {test_loss / len(test_loader):.4f}")
