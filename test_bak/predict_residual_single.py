import sys  
import os 
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))) 
from data_loader import PairedBinaryNumpyDataset
from model import ARCNN
from stats import get_psnr 
import torch 
import argparse 
import glob 
from matplotlib import pyplot as plt 

parser = argparse.ArgumentParser(description='ARCNN Inference on Single File')
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--shape', type=int, nargs='+', required=True)
parser.add_argument('--dtype', type=str, default='f32')
parser.add_argument('--modelpath', type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ARCNN().to(device)
model.load_state_dict(torch.load(args.modelpath, map_location=device))
model.eval()

# Load input and target
shape = tuple(args.shape)
dtype = np.float32 if args.dtype == 'f32' else np.float64
input_array = np.fromfile(args.input, dtype=dtype).reshape(shape)
target_array = np.fromfile(args.target, dtype=dtype).reshape(shape)

# Normalize input to [-1, 1]
x_min = input_array.min()
x_max = input_array.max()
scale = x_max - x_min if x_max != x_min else 1.0
input_norm = ((input_array - x_min) / scale - 0.5) * 2

# Convert to torch tensor
input_tensor = torch.from_numpy(input_norm).unsqueeze(0).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    pred_tensor = model(input_tensor)

# Denormalize output
pred_np = pred_tensor.squeeze().cpu().numpy()
pred_np = 0.5 * (pred_np + 1) * scale + x_min

# Save outputs
# os.makedirs(args.output_dir, exist_ok=True)
# np.save(os.path.join(args.output_dir, "pred.npy"), pred_np.astype(np.float32))
pred_np.astype(np.float32).tofile(os.path.join(args.output_dir, "pred.f32"))
plt.imsave(os.path.join(args.output_dir, "pred.png"), pred_np, cmap='coolwarm')

# Compute PSNRs
psnr_orig = get_psnr(target_array, input_array)
psnr_pred = get_psnr(target_array, pred_np)
print(f"Original PSNR:  {psnr_orig:.2f} dB")
print(f"Corrected PSNR: {psnr_pred:.2f} dB")

# Save PSNRs
# np.savetxt(os.path.join(args.output_dir, "psnr_predict.txt"), [psnr_pred])
# np.savetxt(os.path.join(args.output_dir, "psnr_orig.txt"), [psnr_orig])
