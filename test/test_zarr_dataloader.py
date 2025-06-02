import sys  
import os 
import numpy as np
from torch.utils.data import DataLoader
import glob
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) 
from src.datasets.zarr_loader import ZarrCompressionDataset
from src.utils.stats  import get_psnr, get_ssim 
# from src.utils.stats import get_psnr, get_ssim 


zarr_path = '/lcrc/project/ECP-EZ/jp/git/arcnn/data/hurricane.zarr'


field_names = ['CLOUD','P','PRECIP','QCLOUD',
               'QGRAUP','QICE','QRAIN','QSNOW',
               'QVAPOR','TC','U','V','W']
field_names = ['P']
time_steps = 1
compressors = ['cusz'] 
ebs = ['5e-03']

samples = [] 
for field in field_names:
    for timestep in range(1,time_steps+1):
        for compressor in compressors:
            for eb in ebs:
                samples.append({
                    "field": field,
                    "timestep": timestep,
                    "compressor": compressor,
                    "eb": eb,
                    "z": 0,  # Assuming z=0 for simplicity, adjust as needed
                    "y": 0,  # Assuming y=0 for simplicity, adjust as needed
                    "x": 0   # Assuming x=0 for simplicity, adjust as needed
                })


print(f"Total samples: {len(samples)}") 



dataset = ZarrCompressionDataset(
    zarr_root=zarr_path,
    sample_index=samples,
    patch_size=(64, 64, 64),
    use_original=True
)

# get the first sample fron the dataset 
sample = dataset[0]
print(f"Sample shape: {sample[0].shape}, Target shape: {sample[1].shape}")
# move to CPU and convert to numpy 
sample = (sample[0].cpu().numpy(), sample[1].cpu().numpy())
# Print the shapes of the sample and target
print(f"Sample shape: {sample[0].shape}, Target shape: {sample[1].shape}")
# Calculate PSNR and SSIM
print(f"Sample PSNR: {get_psnr(sample[0][0], sample[1][0])}")
print(f"Sample SSIM: {get_ssim(sample[0][0], sample[1][0])}")
 
# PSNR and SSIM calculations

# Create a DataLoader
