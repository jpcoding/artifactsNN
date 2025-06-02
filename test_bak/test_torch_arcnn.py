# 👇 Choose your device: "cuda" or "cpu"
import torch 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))) 
from torch_arcnn import ARCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# float32 input on device
input32 = torch.randn(1, 1, 2048, 2048, dtype=torch.float32, device=device)

# move model to device
model = ARCNN().to(device=device, dtype=input32.dtype)

# forward pass
out32 = model(input32)
print(out32.shape, out32.dtype, out32.device)
