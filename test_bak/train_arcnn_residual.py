import sys  
import os 
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))) 
from data_loader import BinaryNumpyDataset, PairedBinaryNumpyDataset, compute_data_mean, PairedBinaryNumpyDatasetResidual,PairedBinaryNumpyDatasetResidualE
# from torch_arcnn import ARCNN, ARCNNResidual 
from model import  ARCNN, SRCNN, DnCNN
from torch.utils.data import DataLoader,random_split
import torch 
from torch import nn, optim 
import glob
import argparse

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
parser.add_argument('--savemodel', type=str, help='save the model') 
args = parser.parse_args()
model_str = args.model 
save_model = args.savemodel  
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
# entropy = np.loadtxt(args.entropy)
# threshold = 2.0
# valid_indices = np.where(entropy > threshold)[0]
# sample number 10000 
filtered_input_files = input_files[:] 
filtered_target_files = target_files[:] 

# Split the file lists first
n_total = len(filtered_input_files)
train_len = int(0.8 * n_total)
val_len = int(0.1 * n_total)
test_len = n_total - train_len - val_len

print(f"train_len: {train_len}, val_len: {val_len}, test_len: {test_len}")

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

# Wrap in DataLoaders
n_threads = 8 
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_threads, pin_memory=True, drop_last=True)  
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=n_threads, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_set, batch_size=batch_size ,shuffle=False, num_workers=n_threads, pin_memory=True, drop_last=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ARCNN().to(device)
if model_str == "arcnn":
    model = ARCNN().to(device)
elif model_str == "srcnn":
    model = SRCNN().to(device)
elif model_str == "dncnn":
    model = DnCNN().to(device)
else:
    raise ValueError("Invalid model type. Choose from 'arcnn', 'srcnn', or 'dncnn'.") 
criterion = nn.MSELoss()
# criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=1e-4,          # base_lr from Caffe
#     momentum=0.9,     # momentum from Caffe
#     weight_decay=0.0  # weight_decay from Caffe
# )

def normalize_to_minus1_1_(x, ref_min=None, ref_max=None):
    B = x.size(0)
    if ref_min is None or ref_max is None:
        x_min = x.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        x_max = x.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
    else:
        x_min = ref_min
        x_max = ref_max
    scale = x_max - x_min
    scale[scale == 0] = 1.0
    x.sub_(x_min).div_(scale).mul_(2).sub_(1)
    return x_min, x_max

def denormalize_from_minus1_1_(x, x_min, x_max): 
    x.add_(1).mul_(0.5).mul_(x_max - x_min).add_(x_min)
    return x


def normalize_error(x):
    B = x.size(0)
    x_reshaped = x.view(B, -1)
    abs_max = x_reshaped.abs().max(dim=1)[0].view(B, 1, 1, 1) 
    abs_max[abs_max == 0] = 1.0  # Avoid div by zero 
    return x / abs_max 

def normalize(x, bound):
    bound = bound.view(-1, 1, 1, 1)  # Expand to match x shape
    bound = bound.clamp(min=1e-8)    # Avoid divide-by-zero
    return x / bound

def denormalize(x, bound):
    bound = bound.view(-1, 1, 1, 1)  # Expand to match x shape
    return x * bound

residual_loss_weight = 0.7 
if args.train == "yes":
    best_val_loss = float("inf")
    best_state = None
    num_epochs = 100
    patience = 20
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            residuals = targets - inputs

            # Compute per-sample residual bounds
            residual_bound = residuals.abs().view(residuals.size(0), -1).max(dim=1)[0]  # [B]

            # Normalize residuals
            residuals = normalize(residuals, residual_bound)

            # Normalize inputs (for model input)
            input_min, input_max = normalize_to_minus1_1_(inputs)  # In-place modifies `inputs`
            targets_norm = normalize_to_minus1_1_(targets.clone(), input_min, input_max)  # Avoid in-place for GT

            # Forward pass
            optimizer.zero_grad()
            predicted_residuals = model(inputs)

            # Denormalize predicted residuals
            predicted_residuals_denorm = denormalize(predicted_residuals, residual_bound)
            inputs_denorm = denormalize_from_minus1_1_(inputs.clone(), input_min, input_max)
            predicted_targets = inputs_denorm + predicted_residuals_denorm

            # Dual loss
            
            loss = (
                residual_loss_weight * criterion(predicted_residuals, residuals) +
                (1 - residual_loss_weight) * criterion(predicted_targets, targets)
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                residuals = targets - inputs 
                residual_bound = residuals.abs().view(residuals.size(0), -1).max(dim=1)[0]  
                # Normalize inputs and targets based on input range 
                input_min, input_max = normalize_to_minus1_1_(inputs) 
                # residuals = normalize_error(residuals)
                residuals = normalize(residuals, residual_bound)
                predicted_residuals = model(inputs)
                # Denormalize predicted residuals
                predicted_residuals_denorm = denormalize(predicted_residuals, residual_bound)
                inputs_denorm = denormalize_from_minus1_1_(inputs, input_min, input_max)
                predicted_targets = inputs_denorm + predicted_residuals_denorm
                # Dual loss
                
                loss = (
                    residual_loss_weight * criterion(predicted_residuals, residuals)
                    + (1 - residual_loss_weight) * criterion(predicted_targets, targets)
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break
        # save all the models
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, '{}_epoch_{}.pth'.format(f'cuda_{model_str}', epoch)))

    
    torch.save(best_state, f"best_residual_model_{model_str}_{save_model}.pth")

# ---- TEST ----

model.load_state_dict(torch.load(f"best_residual_model_{model_str}_{save_model}.pth", map_location=device))
model.eval()
test_loss = 0.0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        residuals = targets - inputs 
        residual_bound = residuals.abs().view(residuals.size(0), -1).max(dim=1)[0]  
        # inputs = normalize(inputs, residual_bound)
        residuals = normalize(residuals, residual_bound)

        # Normalize inputs and targets based on input range
        input_min, input_max = normalize_to_minus1_1_(inputs)
        normalize_to_minus1_1_(targets, input_min, input_max)
        # residuals = normalize_error(residuals)
            
        predicted_residuals  = model(inputs)
        # Denormalize predicted residuals
        predicted_residuals_denorm = denormalize(predicted_residuals, residual_bound)
        inputs_denorm = denormalize_from_minus1_1_(inputs, input_min, input_max)
        predicted_targets = inputs_denorm + predicted_residuals_denorm
        # Dual loss
        
        loss = (
            residual_loss_weight * criterion(predicted_residuals, residuals)
            + (1 - residual_loss_weight) * criterion(predicted_targets, targets)
        )
        # loss = criterion(predicted_residuals , residuals)
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader):.6e}")
