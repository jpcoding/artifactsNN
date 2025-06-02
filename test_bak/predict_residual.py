import sys  
import os 
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))) 
from data_loader import PairedBinaryNumpyDatasetResidualE
from torch_arcnn import ARCNNResidual
from stats import get_psnr 
import torch 
import argparse 
import glob 
from matplotlib import pyplot as plt 

parser = argparse.ArgumentParser(description='ARCNNResidual Inference')
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--target_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--shape', type=int, nargs='+', required=True)
parser.add_argument('--dtype', type=str, default='f32')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Load model
model = ARCNNResidual(in_channels=2).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Load dataset
shape = tuple(args.shape) 
dtype = np.float32 if args.dtype == 'f32' else np.float64
input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.f32")))
target_files = sorted(glob.glob(os.path.join(args.target_dir, "*.f32")))
dataset = PairedBinaryNumpyDatasetResidualE(input_files, target_files, shape, dtype=dtype)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

# Create output folders
output_data_dir = os.path.join(args.output_dir, "pred")
output_png_dir = os.path.join(args.output_dir, "pred_png")
os.makedirs(output_data_dir, exist_ok=True)
os.makedirs(output_png_dir, exist_ok=True)

# Inference loop
sample_idx = 0
psnr_list = np.zeros(len(dataset))
orig_psnr = np.zeros(len(dataset))

with torch.no_grad():
    for inputs, residuals, e_scalar in dataloader:
        inputs = inputs.to(device)
        e_scalar = e_scalar.to(device)

        predicted_residuals = model(inputs, e_scalar)
        restored = inputs + predicted_residuals  # final output

        inputs = inputs.cpu()
        restored = restored.cpu()
        targets = (residuals + inputs).cpu()  # reconstruct target = residual + input

        for i in range(restored.size(0)):
            pred_np = restored[i].squeeze(0).numpy()
            target_np = targets[i].squeeze(0).numpy()
            input_np = inputs[i].squeeze(0).numpy()

            # Compute PSNR
            psnr_list[sample_idx] = get_psnr(target_np, pred_np)
            orig_psnr[sample_idx] = get_psnr(target_np, input_np)

            # Save .f32 file
            pred_np.astype(np.float32).tofile(os.path.join(output_data_dir, f"pred_{sample_idx:04d}.f32"))

            # Save PNG
            plt.imsave(os.path.join(output_png_dir, f"pred_{sample_idx:04d}.png"), pred_np, cmap='coolwarm')

            sample_idx += 1

# Save PSNRs
np.savetxt(os.path.join(args.output_dir, "psnr_predict.txt"), psnr_list, fmt='%.6f')
np.savetxt(os.path.join(args.output_dir, "psnr_orig.txt"), orig_psnr, fmt='%.6f')
