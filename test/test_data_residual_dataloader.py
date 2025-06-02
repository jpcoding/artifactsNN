import sys  
import os 
import numpy as np
from torch.utils.data import DataLoader
import glob
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) 
from src.datasets.data_loader import PairedBinaryNumpyDatasetResidualE
from utils.stats import get_psnr



# Example usage of BinaryNumpyDataset
# This is a simple example. You can modify the file_list and shape as per your requirements.
# Command line arguments
parser = argparse.ArgumentParser(description='Binary Numpy Dataset Example')
parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder containing binary files')
parser.add_argument('--target_dir', type=str, required=True, help='Path to the folder containing binary files')
parser.add_argument('--shape', type=int, nargs='+', required=True, help='Shape of the data (e.g., 1 64 64 for RGB)')
parser.add_argument('--dtype', type=str, default='f32', help='Data type (e.g., float32, float64)')
args = parser.parse_args()

# load to cpu 
    
dtype = np.float32 if args.dtype == 'f32' else np.float64 
shape = tuple(args.shape) # 1, W, H 
input_dir = args.input_dir 
target_dir = args.target_dir

input_files = sorted(glob.glob(os.path.join(input_dir, "*.f32")))
target_files = sorted(glob.glob(os.path.join(target_dir, "*.f32")))
dataset = PairedBinaryNumpyDatasetResidualE(input_files, target_files, shape)

dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
# compare data 
psnr_list = np.zeros(len(dataloader.dataset))
sample_idx = 0
for inputs, residuals, scaler in dataloader:
    for i in range(inputs.size(0)):
        input_np = inputs[i].squeeze(0).numpy()
        residual_np = residuals[i].squeeze(0).numpy()
        # scaler_np = scaler[i].to('cpu').numpy()
        cur_psnr = get_psnr(residual_np+input_np, input_np) 
        psnr_list[sample_idx] = cur_psnr
        sample_idx += 1
        
np.savetxt('psnr_list.txt', psnr_list, fmt='%.6f')   


