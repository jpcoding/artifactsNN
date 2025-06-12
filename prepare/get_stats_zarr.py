import zarr 
from concurrent.futures import as_completed, ThreadPoolExecutor 
import numpy as np 
import os 
import sys 
import argparse 
import pandas as pd 


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', type=str, required=True, help='zarr')
parser.add_argument('-o', '--output', type=str, required=True, help='output csv file') 
parser.add_argument('-field', '--field', type=str, required=True, help='field name')  
parser.add_argument('--compression', type=str, default='sz3', help='compression method') 
parser.add_argument('--eb', type=str, default='eb_0.01', help='error bound') 
args = parser.parse_args() 

store = zarr.open(args.inputdir, mode='r', zarr_format=3) 
print(f"Zarr store opened at: {args.inputdir}") 
print("Keys in the store:")
total_steps = 48 
psnrs = np.zeros(total_steps)
ssims = np.zeros(total_steps) 
crs = np.zeros(total_steps) 
for i in range(1, total_steps + 1):
    cur_result = store[f"decompressed/{i}/{args.field}/{args.compression}/{args.eb}"] 
    psnrs[i - 1] = cur_result.attrs.get('psnr', np.nan)
    ssims[i - 1] = cur_result.attrs.get('ssim', np.nan)
    crs[i - 1] = cur_result.attrs.get('cr', np.nan)
results_pd = pd.DataFrame({
    'step': np.arange(1, total_steps + 1),
    'psnr': psnrs,
    'ssim': ssims,
    'cr': crs
})
results_pd.to_csv(args.output, index=False)
print(f"Results saved to {args.output}")
# print(store.tree())
# print(store.tree())

 
