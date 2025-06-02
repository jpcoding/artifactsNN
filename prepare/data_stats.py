import numpy as np 
import os 
import sys 
import argparse 
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor as executor 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/utils/')))

import stats 

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', type=str, required=True, help='input data file')
parser.add_argument('-o', '--outputdir', type=str, required=True, help='output data directory') 
parser.add_argument('-d', '--dtype', type=str, default='f32', help='data type')
parser.add_argument('-n', '--num', type=int, default=1, help='dimension of data')
parser.add_argument('-s', '--shape', nargs='+', type=int, default=1, help='shape of data')
parser.add_argument('--ext', type=str, default='.f32', help='extension of the data files')

args = parser.parse_args() 
numpy_data_type = np.float32 if args.dtype == 'f32' else np.float64
# get a list of all the files in the input directory
input_files = [f for f in os.listdir(args.inputdir) if f.endswith(args.ext)]
input_files.sort()

print(input_files) 
psnr_list = np.zeros(len(input_files))   
ssim_list = np.zeros(len(input_files)) 
error_std = np.zeros(len(input_files)) 
data_std = np.zeros(len(input_files))   
data_mean  = np.zeros(len(input_files)) 
error_mean = np.zeros(len(input_files)) 

# for i,  inputfile in  enumerate(input_files):
#     in_path = os.path.join(args.inputdir, inputfile) 
#     out_path = os.path.join(args.outputdir, inputfile)
#     psnr_list[i] = stats.get_psnr(in_path, out_path) 
#     ssim_list[i] = stats.get_ssim(in_path, out_path) 

def process_file(i):
    in_path = os.path.join(args.inputdir, input_files[i]) 
    out_path = os.path.join(args.outputdir, input_files[i])
    in_data = np.fromfile(in_path, dtype=numpy_data_type).reshape(args.shape) 
    out_data = np.fromfile(out_path, dtype=numpy_data_type).reshape(args.shape) 
    psnr = stats.get_psnr(in_data, out_data) 
    ssim = stats.qcatssim(in_data, out_data) 
    psnr_list[i] = psnr
    ssim_list[i] = ssim
    error_std[i] = np.std(in_data - out_data) 
    data_std[i] = np.std(in_data)
    data_mean[i] = np.mean(in_data) 
    error_mean[i] = np.mean(in_data - out_data) 

# parallel processing   
with executor(max_workers=64) as ex:
    ex.map(process_file, range(len(input_files)))
# Loop through each file and process it
# for inputfile in fildterd_files:
    
results_pd = pd.DataFrame({'file':input_files, 
                           'psnr': psnr_list, 
                           'ssim': ssim_list,
                           'error_std': error_std,
                            'error_mean': error_mean, 
                           'data_std': data_std,
                           'data_mean': data_mean})   
results_pd.to_csv('stats.csv', index=False) 
    
    



