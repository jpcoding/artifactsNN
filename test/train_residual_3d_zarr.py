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
from src.datasets.zarr_loader import ZarrCompressionDataset



# Example usage of BinaryNumpyDataset
# This is a simple example. You can modify the file_list and shape as per your requirements.
# Command line arguments
parser = argparse.ArgumentParser(description='Binary Numpy Dataset Example')
parser.add_argument('--data_dir', type=str, required=True, help='Zarr data directory containing compressed files') 
parser.add_argument('--shape', type=int, nargs='+', required=True, help='Shape of the original data ')
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


zarr_path = '/lcrc/project/ECP-EZ/jp/git/arcnn/data/hurricane.zarr'

field_names = ['P']

compressors = ['cusz'] 
ebs = ['5e-03']

data_shape = [100,500,500] # z, y, x 
stride = 32 
# on a field cretate a list of samples using stride and shape 
shape_samples = [] 
for i in range(0, data_shape[0] - shape[0] + 1, stride): 
    for j in range(0, data_shape[1] - shape[1] + 1, stride):
        for k in range(0, data_shape[2] - shape[2] + 1, stride):
            shape_samples.append([i, j, k]) 
# shape_samples = shape_samples[0:10]

print("num of samples: ", len(shape_samples)) 

samples = [] 
train_start_time = 1
train_end_time = 8 
for field in field_names:
    for timestep in range(train_start_time, train_end_time+1): 
        for compressor in compressors:
            for eb in ebs:
                for z, y, x in shape_samples: 
                    samples.append({
                        "field": field,
                        "timestep": timestep,
                        "compressor": compressor,
                        "eb": eb,
                        "z": z,  
                        "y": y, 
                        "x": x   
                    })
                

train_set = ZarrCompressionDataset(
    zarr_root=zarr_path,
    sample_index=samples,
    patch_size=shape,
    use_original=True
)

samples = [] 
val_start_time = 9
val_end_time = 10 
for field in field_names:
    for timestep in range(val_start_time, val_end_time+1):
        for compressor in compressors:
            for eb in ebs:
                for z, y, x in shape_samples: 
                    samples.append({
                        "field": field,
                        "timestep": timestep,
                        "compressor": compressor,
                        "eb": eb,
                        "z": z,  
                        "y": y,  
                        "x": x   
                    })
val_set = ZarrCompressionDataset(
    zarr_root=zarr_path,
    sample_index=samples,
    patch_size=shape,
    use_original=True
)

train_loader = DataLoader(
    train_set, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)
val_loader = DataLoader(
    val_set, 
    batch_size=args.batch_size, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

print(f"Total training samples: {len(train_set)}") 
print(f"Total validation samples: {len(val_set)}")


    
# exit()

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
    

# # ---- TEST ----
# test_config = SimpleNamespace(  
#     model=model,
#     criterion=criterion,
#     test_loader=test_loader,
#     device=device,
#     model_str=model_str,
#     checkpoint_path = bestpath,
# ) 
# test_residual_3d(test_config) 
