import numpy as np 
import os 
import sys 
import argparse 
from matplotlib import pyplot as plt 
from skimage.util import view_as_windows
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from stats import compute_patch_psnr_batch

def get_psnr(src_data, dec_data):

    data_range = np.max(src_data) - np.min(src_data)
    diff = src_data - dec_data
    max_diff = np.max(abs(diff))
    # print("abs err={:.8G}".format(max_diff))
    mse = np.mean(diff ** 2)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr 

def quantization(data: np.ndarray, recip_precision: float):
    quant_idx = np.round(data * recip_precision).astype(np.int32)    
    quantized_data = (quant_idx / recip_precision).astype(data.dtype) 
    data_max = np.max(data) 
    data_min = np.min(data) 
    quantized_data = np.clip(quantized_data, data_min, data_max)
    return quantized_data, quant_idx.astype(np.int32) 


def entropy(data: np.ndarray):
    _, counts = np.unique(data, return_counts=True) 
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]  # Remove zero probabilities
    return -np.sum(probs * np.log2(probs))



def extract_slice_fast_stacked(input: np.ndarray, target: np.ndarray, patch_size: int, stride: int, return_psnr: bool = False,
                               normalize: bool = True): 
    """
    Efficiently extracts and normalizes patches from input and target slices using sliding windows.
    Returns stacked arrays: [N, patch_size, patch_size]
    """
    input_patches = view_as_windows(input, (patch_size, patch_size), step=stride)
    target_patches = view_as_windows(target, (patch_size, patch_size), step=stride)
    n_x, n_y = input_patches.shape[:2]
    input_patches = input_patches.reshape(-1, patch_size, patch_size)
    target_patches = target_patches.reshape(-1, patch_size, patch_size)
    if normalize:
    # Compute min/max per patch (axis=1,2)
        p_min = input_patches.reshape(len(input_patches), -1).min(axis=1, keepdims=True)
        p_max = input_patches.reshape(len(input_patches), -1).max(axis=1, keepdims=True)
        scale = p_max - p_min
        scale[scale == 0] = 1.0
        # Normalize to [-1, 1]
        input_patches = ((input_patches.reshape(len(input_patches), -1) - p_min) / scale - 0.5) * 2
        target_patches = ((target_patches.reshape(len(target_patches), -1) - p_min) / scale - 0.5) * 2

        # Reshape back to [N, patch_size, patch_size]
        input_patches = input_patches.reshape(-1, patch_size, patch_size)
        target_patches = target_patches.reshape(-1, patch_size, patch_size)
    
    psnr_values = None
    if return_psnr:
        psnr_values = compute_patch_psnr_batch(input_patches, target_patches)


    return input_patches, target_patches, psnr_values 


def save_patches(input_patches, target_patches, output_dir, prefix):
    """
    Saves the extracted patches to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    subfolder_name = ['input', 'target', 'input_png', 'target_png'] 
    for subfolder in subfolder_name:
        subfolder_path = os.path.join(output_dir, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    for i, (input_patch, target_patch) in enumerate(zip(input_patches, target_patches)):
        input_patch.astype(np.float32).tofile(os.path.join(output_dir, f"input/{prefix}_{i:04d}.f32"))
        target_patch.astype(np.float32).tofile(os.path.join(output_dir, f"target/{prefix}_{i:04d}.f32"))
        plt.imsave(os.path.join(output_dir, f"input_png/{prefix}_{i:04d}.png"), input_patch, cmap='coolwarm')
        plt.imsave(os.path.join(output_dir, f"target_png/{prefix}_{i:04d}.png"), target_patch, cmap='coolwarm')


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
parser.add_argument('-p', '--patch_size', type=int, default=64, help='patch size')
parser.add_argument('-st', '--stride', type=int, default=32, help='stride size')
parser.add_argument('-t', '--entropy_threshold', type=float, default=2.0, help='entropy threshold')
parser.add_argument('--normalize', action='store_true', help='normalize patches') 



args = parser.parse_args()
print("Data normalization: ", args.normalize)


numpy_data_type = np.float32 if args.dtype == 'f32' else np.float64 


orig_data = np.fromfile(args.input, dtype=numpy_data_type).reshape(args.shape)  
precision = 1.0 /(2* args.eb) 
quantized_data, quant_idx = quantization(orig_data, precision)

output_folder = args.output 

if not os.path.exists(args.output):
    os.makedirs(args.output)
subfolder_name = os.path.basename(args.input)
# creat subfolder 

        
entropy_threshold = 2.0 

# save quantized data all the slices using the following indexex 

x1 = 0
y1 = 0 
x2 = -1
y2 = -1

slices_start = 0
slices_end =  args.shape[0] - 1 
slice_entropys = np.zeros(slices_end - slices_start + 1)
psnr_lists = np.zeros(slices_end - slices_start + 1)
patch_psnr_values = None
for i in range(slices_start, slices_end + 1):
    cur_entropy = entropy(quantized_data[i, x1:x2, y1:y2])   
    slice_entropys[i] = cur_entropy
    if cur_entropy < entropy_threshold:
        print(f"slice {i} has low entropy: {cur_entropy}")
        continue 
    else:
        input_patches, target_patches, patch_psnr_values = extract_slice_fast_stacked(quantized_data[i, x1:x2, y1:y2], 
                                                                                orig_data[i, x1:x2, y1:y2],
                                                                                64, 32, return_psnr=True, normalize=args.normalize) 
        print("number of patches: ", len(input_patches)) 
        save_patches(input_patches, target_patches, output_folder, subfolder_name+f'_slice_{i:04d}')  

    psnr_lists[i] = get_psnr(orig_data[i, x1:x2, y1:y2], quantized_data[i, x1:x2, y1:y2])
# save entropy of the data
entropy_file = os.path.join(output_folder, f'{subfolder_name}_slice_entropy.txt')
np.savetxt(entropy_file, slice_entropys, fmt='%f') 
# save psnr of the data
psnr_file = os.path.join(output_folder, f'{subfolder_name}_slice_psnr.txt')
np.savetxt(psnr_file, psnr_lists, fmt='%f')
    
    
