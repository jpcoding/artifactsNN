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

parser = argparse.ArgumentParser(description='ARCNN Inference')
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--target_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--shape', type=int, nargs='+', required=True)
parser.add_argument('--dtype', type=str, default='f32')
parser.add_argument('--modelpath', type=str, required=True, help='model path')  
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Load model
model = ARCNN().to(device)
model.load_state_dict(torch.load(args.modelpath, map_location=device))
model.eval()

# Load dataset
shape = tuple(args.shape) 
dtype = np.float32 if args.dtype == 'f32' else np.float64
input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.f32")))
target_files = sorted(glob.glob(os.path.join(args.target_dir, "*.f32")))
dataset = PairedBinaryNumpyDataset(input_files, target_files, shape, dtype=dtype)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=8)

# Create output folders
output_data_dir = os.path.join(args.output_dir, "pred")
output_png_dir = os.path.join(args.output_dir, "pred_png")
os.makedirs(output_data_dir, exist_ok=True)
os.makedirs(output_png_dir, exist_ok=True)

# Inference loop
sample_idx = 0
psnr_list = np.zeros(len(dataset))
orig_psnr = np.zeros(len(dataset))


def normalize_to_minus1_1_(x, ref_min=None, ref_max=None):
    B = x.size(0)
    if ref_min is None or ref_max is None:
        x_min = x.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        x_max = x.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
    else:
        x_min = ref_min
        x_max = ref_max
    scale = x_max - x_min
    scale[scale == 0] = 1.0
    x.sub_(x_min).div_(scale).mul_(2).sub_(1)
    return x_min, x_max



def denormalize_from_minus1_1_(x, x_min, x_max):
    x.add_(1).mul_(0.5).mul_(x_max - x_min).add_(x_min)
    return x


with torch.no_grad():
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        # normalize inputs 
        x_min, x_max = normalize_to_minus1_1_(inputs) 
        predicted_targets  = model(inputs)
        # denormalize inputs
        inputs = denormalize_from_minus1_1_(inputs, x_min, x_max)
        predicted_targets = denormalize_from_minus1_1_(predicted_targets, x_min, x_max) 
        
        inputs = inputs.cpu()
        predicted_targets = predicted_targets.cpu()
        for i in range(predicted_targets.size(0)):
            pred_np = predicted_targets[i].squeeze(0).numpy()
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
