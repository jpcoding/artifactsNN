import sys  
import os 
import numpy as np
import torch 
from torch import nn, optim 
import glob 
import argparse
from types import SimpleNamespace
from torch.utils.data import DataLoader,random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) 
from src.datasets.data_loader import PairedBinaryNumpyDataset
from src.models.model import ARCNN, SRCNN, DnCNN3D, UNet3D 
from src.training.train_model import train_residual_3d
from src.training.test_model  import test_residual_3d



# Example usage of BinaryNumpyDataset
# This is a simple example. You can modify the file_list and shape as per your requirements.
# Command line arguments
parser = argparse.ArgumentParser(description='Binary Numpy Dataset Example')
parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder containing binary files')
parser.add_argument('--target_dir', type=str, required=True, help='Path to the folder containing binary files')
parser.add_argument('--shape', type=int, nargs='+', required=True, help='Shape of the data (e.g., 1 64 64 for RGB)')
parser.add_argument('--dtype', type=str, default='f32', help='Data type (e.g., float32, float64)')
parser.add_argument('--entropy', type=str, required=True, help='entropy of the data') 
parser.add_argument('--train', type=str, default='yes', help='train the model')
parser.add_argument('--outputs_dir', type=str, default='outputs', help='output directory') 
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer') 
parser.add_argument('--model', type=str, default='arcnn', help='model to use (arcnn, srcnn, dncnn)') 
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training and testing') 
parser.add_argument('--bestpath', type=str, help='save the model') 
parser.add_argument('--range_penalty', type=float, default=0.0, help='range penalty')
parser.add_argument('--num_train', type=int, default=1000, help='number of training samples') 
parser.add_argument('--penalty_fn', type=str, default='range_penalty', help='penalty function')
parser.add_argument('--penalty_order', type=float, default=2., help='penalty order')  
parser.add_argument('--load_from_file', type=str, default='yes', help='load from file') 
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
input_dir = args.input_dir  
target_dir = args.target_dir
batch_size = args.batch_size 

input_files = sorted(glob.glob(os.path.join(input_dir, "*.f32")))
target_files = sorted(glob.glob(os.path.join(target_dir, "*.f32")))

# shuffle the input and target files     

# input_files = np.random.permutation(input_files)


# entropy = np.loadtxt(args.entropy)
# threshold = 2.0
# valid_indices = np.where(entropy > threshold)[0]
# sample number 10000 

if args.load_from_file == "yes":  
    total_len = len(input_files) 
    use_len = args.num_train 
    if use_len > total_len:
        use_len = total_len 
    selected_indices = np.random.choice(len(input_files), use_len, replace=False)
    filtered_input_files = [input_files[i] for i in selected_indices]
    filtered_target_files = [target_files[i] for i in selected_indices]

    # filtered_input_files = input_files
    # filtered_target_files = target_files
    # Split the file lists first
    n_total = len(filtered_input_files)
    train_len = int(0.8 * n_total)
    val_len = int(0.1 * n_total)
    test_len = n_total - train_len - val_len
    # print(f"train_len: {train_len}, val_len: {val_len}, test_len: {test_len}")
    train_input_files = filtered_input_files[:train_len]
    train_target_files = filtered_target_files[:train_len]
    val_input_files = filtered_input_files[train_len:train_len+val_len]
    val_target_files = filtered_target_files[train_len:train_len+val_len]
    test_input_files = filtered_input_files[train_len+val_len:]
    test_target_files = filtered_target_files[train_len+val_len:]
    # Compute training mean
    # Create datasets with the computed mean
    train_set = PairedBinaryNumpyDataset(train_input_files, train_target_files, shape)
    val_set   = PairedBinaryNumpyDataset(val_input_files, val_target_files, shape)
    test_set  = PairedBinaryNumpyDataset(test_input_files, test_target_files, shape)
    print(f"train_len: {len(train_set)}, val_len: {len(val_set)}, test_len: {len(test_set)}") 
    # Wrap in DataLoaders
    n_threads = 8 
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_threads, pin_memory=True, drop_last=True)  
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=n_threads, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size ,shuffle=False, num_workers=n_threads, pin_memory=True, drop_last=False)

else:
    # load from memory
    # the data is not divided into chunks yet, the data loader will have to do that here
    # there are 10 time steps, each time step is 100x500x500, 
    # use the first 8 time steps for training, the last 2 time steps for validation set 
    pass 

    


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
test_config = SimpleNamespace(  
    model=model,
    criterion=criterion,
    test_loader=test_loader,
    device=device,
    model_str=model_str,
    checkpoint_path = bestpath,
) 
test_residual_3d(test_config) 
