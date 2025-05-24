import os 
import sys 
import numpy as np 
import argparse    
import shutil


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
    decompressed = (quant_idx / recip_precision).astype(data.dtype) 
    return decompressed, quant_idx.astype(np.int32) 


def entropy(data: np.ndarray):
    _, counts = np.unique(data, return_counts=True) 
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]  # Remove zero probabilities
    return -np.sum(probs * np.log2(probs))


parser = argparse.ArgumentParser(
                    prog='prepare_data',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-i', '--inputfile', type=str, required=True, help='input data file') 
parser.add_argument('-o', '--outputdir', type=str, required=True, help='output data directory')
parser.add_argument('-e', '--eb', type=float, required=True, help='error bound')  
parser.add_argument('-d', '--dtype', type=str, default='f32', help='data type')
parser.add_argument('-n', '--num', type=int, default=1, help='dimension of data') 
parser.add_argument('-s', '--shape', nargs='+', type=int, default=1, help='shape of data')
parser.add_argument("--artifact", type=str, default='banding', help='artifacts type') 
parser.add_argument('--use', type=str, default='train', help='prefix of the data folder') 

args = parser.parse_args()
numpy_data_type = np.float32 if args.dtype == 'f32' else np.float64 
artifact = args.artifact 

block_size = 64 
step = 32

rel_eb = args.eb 
original_data = np.fromfile(args.inputfile, dtype=numpy_data_type).reshape(args.shape)       
abs_eb = (original_data.max() - original_data.min()) * rel_eb 
decompressed , quant_idx = quantization(original_data, 1.0 / (2 * args.eb)) 

level1_names = ['orig', 'decompressed']
os.makedirs(args.outputdir, exist_ok=True)   
for level1 in level1_names:
    os.makedirs(os.path.join(args.outputdir, level1), exist_ok=True) 
    if(level1 =='decompressed'):
        level2 = artifact # artifatc name
        os.makedirs(os.path.join(args.outputdir, level1, level2), exist_ok=True)
        level3 = str(args.eb) # error bound 
        os.makedirs(os.path.join(args.outputdir, level1, level2, level3), exist_ok=True)
        level4 = args.use  # usage type, e.g., trai or val 
        os.makedirs(os.path.join(args.outputdir, level1, level2, level3, level4), exist_ok=True) 
        # save the decompressed data 
        decompressed.tofile(os.path.join(args.outputdir, level1, level2, level3, level4, os.path.basename(args.inputfile)))  
    else: 
        # save a copy of the original data
        level2 = args.use  
        os.makedirs(os.path.join(args.outputdir, level1, level2), exist_ok=True) 
        shutil.copy(args.inputfile, os.path.join(args.outputdir, level1, level2,  os.path.basename(args.inputfile))) 




        
    









