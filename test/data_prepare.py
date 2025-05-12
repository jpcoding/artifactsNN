import numpy as np 
import os 
import sys 
import argparse 
from matplotlib import pyplot as plt 

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



args = parser.parse_args()

numpy_data_type = np.float32 if args.dtype == 'f32' else np.float64 


orig_data = np.fromfile(args.input, dtype=numpy_data_type).reshape(args.shape)  
precision = 1.0 /(2* args.eb) 
quantized_data, quant_idx = quantization(orig_data, precision)

if not os.path.exists(args.output):
    os.makedirs(args.output)
subfolder_name = os.path.basename(args.input)
# creat subfolder 
subfolder_path = os.path.join(args.output, subfolder_name)
if not os.path.exists(subfolder_path):
    os.makedirs(subfolder_path)
    
subsubfolders = ['quantized_data', 'orig_data', 'orig_png','quantized_png'] 
for subsubfolder in subsubfolders:
    subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
    if not os.path.exists(subsubfolder_path):
        os.makedirs(subsubfolder_path)
        

# save quantized data all the slices using the following indexex 

x1 = 0
y1 = 0 
x2 = 200
y2 = 200 

slices_start = 0
slices_end =  args.shape[0] - 1 
slice_entropys = np.zeros(slices_end - slices_start + 1)

psnr_lists = np.zeros(slices_end - slices_start + 1) 

for i in range(slices_start, slices_end + 1):
    cur_entropy = entropy(quantized_data[i, x1:x2, y1:y2])   
    
    slice_entropys[i] = cur_entropy
    quantized_data[i, x1:x2, y1:y2].astype(numpy_data_type).tofile(os.path.join(subfolder_path, f'quantized_data/slice_{i:04d}.f32'))
    orig_data[i, x1:x2, y1:y2].astype(numpy_data_type).tofile(os.path.join(subfolder_path, f'orig_data/slice_{i:04d}.f32')) 
    # save the quantized data as image
    plt.imsave(os.path.join(subfolder_path, f'orig_png/slice_{i:04d}.png'), orig_data[i, x1:x2, y1:y2], cmap='coolwarm') 
    plt.imsave(os.path.join(subfolder_path, f'quantized_png/slice_{i:04d}.png'), quantized_data[i, x1:x2, y1:y2], cmap='coolwarm') 
    psnr_lists[i] = get_psnr(orig_data[i, x1:x2, y1:y2], quantized_data[i, x1:x2, y1:y2])
# save entropy of the data
entropy_file = os.path.join(subfolder_path, 'entropy.txt')
np.savetxt(entropy_file, slice_entropys, fmt='%f') 
# save psnr of the data
psnr_file = os.path.join(subfolder_path, 'psnr.txt')
np.savetxt(psnr_file, psnr_lists, fmt='%f')
    
    
