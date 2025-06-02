import numpy as np 
import os 
import sys 
import argparse 
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial 

def get_psnr(src_data, dec_data):

    data_range = np.max(src_data) - np.min(src_data)
    diff = src_data - dec_data
    max_diff = np.max(abs(diff))
    print("abs err={:.8G}".format(max_diff))
    mse = np.mean(diff ** 2)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr 

def quantization(data: np.ndarray, recip_precision: float):
    quant_idx = np.round(data * recip_precision).astype(np.int32)    
    quantized_data = (quant_idx / recip_precision).astype(data.dtype) 
    return quantized_data, quant_idx.astype(np.int32) 


def entropy(data: np.ndarray):
    _, counts = np.unique(data, return_counts=True) 
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]  # Remove zero probabilities
    return -np.sum(probs * np.log2(probs))




parser = argparse.ArgumentParser(
                    prog='prepare_data',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-i', '--input', type=str, required=True, help='input data file') 
parser.add_argument('-o', '--output', type=str, required=True, help='output data directory')
parser.add_argument('-e', '--eb', type=float, required=True, help='error bound')  
parser.add_argument('-d', '--dtype', type=str, default='f32', help='data type')
parser.add_argument('-n', '--num', type=int, default=1, help='dimension of data') 
parser.add_argument('-s', '--shape', nargs='+', type=int, default=1, help='shape of data')
parser.add_argument('--prefix', type=str, default='data', help='prefix of the data folder') 



args = parser.parse_args()

numpy_data_type = np.float32 if args.dtype == 'f32' else np.float64 


orig_data = np.fromfile(args.input, dtype=numpy_data_type).reshape(args.shape)  
rel_eb = args.eb 
abs_eb = (orig_data.max() - orig_data.min()) * rel_eb
precision = 1.0 /(2* abs_eb) 
quantized_data, quant_idx = quantization(orig_data, precision)

if not os.path.exists(args.output):
    os.makedirs(args.output)
subfolder_name = os.path.basename(args.input)
subfolder_name = args.prefix +'_' + subfolder_name
# creat subfolder 
subfolder_path = os.path.join(args.output, subfolder_name)
if not os.path.exists(subfolder_path):
    os.makedirs(subfolder_path)
    
    
subsubfolders = ['quantized_data', 'orig_data'] 
for subsubfolder in subsubfolders:
    subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
    if not os.path.exists(subsubfolder_path):
        os.makedirs(subsubfolder_path)
    
# save quantized data all the slices using the following indexex 

block_size = 64 
stride = 32

# sliding window., check the entropy of each window, 

base_name = os.path.basename(args.input)

def process_block(x1, y1, z1, block_size, shape, quant_idx, quantized_data, orig_data, numpy_data_type, subsubfolder_path):
    x2 = min(x1 + block_size, shape[0])
    y2 = min(y1 + block_size, shape[1])
    z2 = min(z1 + block_size, shape[2])

    if x2 - x1 != block_size or y2 - y1 != block_size or z2 - z1 != block_size:
        return  # Skip incomplete block

    block_qidx = quant_idx[x1:x2, y1:y2, z1:z2]
    block_quant = quantized_data[x1:x2, y1:y2, z1:z2]
    block_orig = orig_data[x1:x2, y1:y2, z1:z2]

    entropy_val = entropy(block_qidx)

    quantized_data_path = os.path.join(subfolder_path, f'quantized_data/data_{x1}_{y1}_{z1}_{entropy_val:.3f}.f32')
    orig_data_path = os.path.join(subfolder_path, f'orig_data/data_{x1}_{y1}_{z1}_{entropy_val:.3f}.f32')
    print(f"Saving quantized data to {quantized_data_path}")

    block_quant.astype(numpy_data_type).tofile(quantized_data_path)
    block_orig.astype(numpy_data_type).tofile(orig_data_path)
    print(f"Processed block at ({x1}, {y1}, {z1}) with entropy {entropy_val:.3f}")



shape = args.shape 
tasks = []
for i in range(0, shape[0], stride):
    for j in range(0, shape[1], stride):
        for k in range(0, shape[2], stride):
            x2 = min(i + block_size, shape[0])
            y2 = min(j + block_size, shape[1])
            z2 = min(k + block_size, shape[2])
            if x2 - i == block_size and y2 - j == block_size and z2 - k == block_size:
                tasks.append((i, j, k))

# Parallel processing
print(tasks  )
with ProcessPoolExecutor() as executor:
    for i, j, k in tasks:
        process_block(i, j, k, block_size, shape, quant_idx, 
                      quantized_data, orig_data, 
                      numpy_data_type, subsubfolder_path)