import sys  
import os 
import numpy as np
from torch.utils.data import DataLoader
import torch 
from torch import nn, optim 
import glob 
import argparse
from types import SimpleNamespace
from torch.utils.data import DataLoader, Subset, RandomSampler
import pandas as pd 
from tqdm import tqdm
import re 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) 
from src.datasets.data_loader import PairedBlockDataset 
from src.models.model import ARCNN, SRCNN, DnCNN3D, UNet3D 
from src.training.train_model import train_residual_3d 
from src.training.test_model  import test_residual_3d 




# Example usage of BinaryNumpyDataset
# This is a simple example. You can modify the file_list and shape as per your requirements.
# Command line arguments
parser = argparse.ArgumentParser(description='Binary Numpy Dataset Example')
parser.add_argument('--train_input', type=str, required=True, help='Path to the folder containing binary files') 
parser.add_argument('--train_target', type=str, required=True, help='Path to the folder containing binary files') 
parser.add_argument('--val_input', type=str, required=True, help='Path to the folder containing binary files')
parser.add_argument('--val_target', type=str, required=True, help='Path to the folder containing binary files')
parser.add_argument('--test_input', type=str, required=False, help='Path to the folder containing binary files') 
parser.add_argument('--test_target', type=str, required=False, help='Path to the folder containing binary files')

parser.add_argument('--shape', type=int, nargs='+', required=True, help='Shape of the input chuncks')
parser.add_argument('--dtype', type=str, default='f32', help='Data type (e.g., float32, float64)')
parser.add_argument('--train', type=str, default='yes', help='train the model')
parser.add_argument('--outputs_dir', type=str, default='outputs', help='output directory') 
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer') 
parser.add_argument('--model', type=str, default='arcnn', help='model to use (arcnn, srcnn, dncnn)') 
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training and testing') 
parser.add_argument('--bestpath', type=str, help='save the model') 
parser.add_argument('--range_penalty', type=float, default=0.0, help='range penalty')
parser.add_argument('--penalty_fn', type=str, default='range_penalty', help='penalty function')
parser.add_argument('--penalty_order', type=float, default=2., help='penalty order')  
parser.add_argument('--num_epoches', type=int, default=100, help='number of epochs')

args = parser.parse_args()
model_str = args.model 
bestpath = args.bestpath  
if not os.path.exists(args.outputs_dir): 
    os.makedirs(args.outputs_dir) 

print(args.train)

torch.manual_seed(123)

# load to cpu 
    
dtype = np.float32 if args.dtype == 'f32' else np.float64 
shape = tuple(args.shape)
batch_size = args.batch_size 


def select_fields(file_list, field_names):
    pattern = r"^([A-Z0-9]+)f(\d{2})\.bin\.f32$"
    selected_files = []
    for file in file_list:
        basename = os.path.basename(file)
        match = re.match(pattern, basename)  
        for field_name in field_names:
            if match.group(1) == field_name: 
                selected_files.append(file)
    return selected_files

# pick_fields = ["U", "V", "W"]

pick_fields = ["U"]
timesteps = slice(0,1)
# pick_fields = ["U", "V", "W", "TKE", "EPSILON", "OMEGA"] 
train_input_files = sorted(glob.glob(os.path.join(args.train_input, "*.f32")))
train_target_files = sorted(glob.glob(os.path.join(args.train_target, "*.f32")))
val_input_files = sorted(glob.glob(os.path.join(args.val_input, "*.f32")))
val_target_files = sorted(glob.glob(os.path.join(args.val_target, "*.f32")))

train_input_files = select_fields(train_input_files, pick_fields    )[timesteps]
train_target_files = select_fields(train_target_files, pick_fields   )[timesteps]
val_input_files = select_fields(val_input_files, pick_fields     )[timesteps]
val_target_files = select_fields(val_target_files, pick_fields   )[timesteps]


if args.test_input is not None and args.test_target is not None: 
    test_input_files = sorted(glob.glob(os.path.join(args.test_input, "*.f32")))
    test_target_files = sorted(glob.glob(os.path.join(args.test_target, "*.f32")))
    test_input_files = select_fields(test_input_files, pick_fields   )[timesteps]
    test_target_files = select_fields(test_target_files, pick_fields     )[timesteps]
else:
    test_input_files = None
    test_target_files = None


# print("test_input_files", test_input_files)
# print("test_target_files", test_target_files)
print("train_input_files", len(train_input_files))
print("train_target_files", len(train_target_files))
print("val_input_files", val_input_files)
print("val_target_files", val_target_files)

# created paired dataset
train_dataset = PairedBlockDataset(train_input_files, train_target_files, shape, 
                                   block_size = 64, stride = 32 , dtype=dtype)
val_dataset = PairedBlockDataset(val_input_files, val_target_files, shape, 
                                 block_size = 64, stride = 32 , dtype=dtype)
if test_input_files is not None and test_target_files is not None:
    test_dataset = PairedBlockDataset(test_input_files, test_target_files, shape, 
                                      block_size = 64, stride = 32 , dtype=dtype)
    
nthreads = 16 
 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=nthreads, pin_memory=True, drop_last=True) 
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=nthreads, pin_memory=True, drop_last=False)
if test_input_files is not None and test_target_files is not None:
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    

    
#  -- test the data loader  -- 

if False:  
    chunck_mse = []
    chunck_err_std = [] 
    chunck_data_range = []   
    chunck_data_std = [] 


    effective_chunks = 0
    for _, _ in train_loader: # Iterate to count, no heavy processing
        effective_chunks += _.shape[0] # Get batch size of current batch
        
    print("effective_chunks", effective_chunks) 
    # -- compute the stats of the chuncks -- 
    chunck_mse = np.zeros(effective_chunks)
    chunck_data_std = np.zeros(effective_chunks) 
    chunck_data_range = np.zeros(effective_chunks) 
    chunck_err_std = np.zeros(effective_chunks) 

    cur_inx = 0 


    for i, (input_data, target_data) in enumerate(tqdm(train_loader, desc="Computing stats")):
        input_data = input_data.numpy() 
        target_data = target_data.numpy() 
        print("input_data", input_data.shape) 
        actual_batch_szie = input_data.shape[0] 
        # print(input_data.shape) 
        # print(target_data.shape) 
        # exit(0) 
        # compute the stats of the chuncks
        batch_mse = np.mean((input_data - target_data) ** 2, axis=(1, 2, 3,4)) 
        batch_mean = np.mean(input_data, axis=(1, 2, 3,4)) 
        batch_data_std = np.std(input_data, axis=(1, 2, 3,4)) 
        batch_data_range = np.max(input_data, axis=(1, 2, 3,4)) - np.min(input_data, axis=(1, 2, 3,4))
        batch_err_std = np.std(input_data - target_data, axis=(1, 2, 3,4)) 
        chunck_mse[cur_inx:cur_inx + actual_batch_szie] = batch_mse
        chunck_data_std[cur_inx:cur_inx + actual_batch_szie] = batch_data_std
        chunck_data_range[cur_inx:cur_inx + actual_batch_szie] = batch_data_range
        chunck_err_std[cur_inx:cur_inx + actual_batch_szie] = batch_err_std
        cur_inx += actual_batch_szie 


# save the stats to a csv file 
    stats_df = pd.DataFrame({
        'mse': chunck_mse,
        'data_std': chunck_data_std,
        'data_range': chunck_data_range,
        'err_std': chunck_err_std,
    }) 
    stats_df.to_csv(os.path.join(args.outputs_dir, 'train_stats.csv'), index=False) 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ARCNN().to(device)
if model_str == "arcnn":
    model = ARCNN().to(device)
elif model_str == "srcnn":
    model = SRCNN().to(device)
elif model_str == "dncnn":
    model = DnCNN3D().to(device)
elif model_str == "unet":
    model = UNet3D().to(device) 
else:
    raise ValueError("Invalid model type. Choose from 'arcnn', 'srcnn', or 'dncnn'.") 


optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
criterion = nn.MSELoss()

traning_config = SimpleNamespace(
    optimizer=optimizer, 
    criterion=criterion, 
    train_loader=train_loader, 
    val_loader=val_loader,
    device=device,
    model=model,
    model_str=model_str,
    num_epochs=args.num_epoches,
    patience=20,
    lambda_range=args.range_penalty, 
    best_model_path=bestpath, 
    outputs_dir=args.outputs_dir,
    range_penalty_order=args.penalty_order, 
    penalty_fn=args.penalty_fn, 
)

# ---- TRAIN ----
if args.train == "yes":
    train_residual_3d(traning_config) 
    

# ---- TEST ----

# test_config = SimpleNamespace(  
#     model=model,
#     criterion=criterion,
#     test_loader=test_loader,
#     device=device,
#     model_str=model_str,
#     checkpoint_path = bestpath,
# ) 
# test_dncnn3d(test_config) 
