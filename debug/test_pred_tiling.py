
## residual model 
import numpy as np 
from matplotlib import pyplot as plt
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('./'), '../'))) 

from src.models import  SRCNN, ARCNN, DnCNN3D, UNet3D
from src.training import train_model, test_model 
from src.utils.stats import get_psnr, qcatssim
from src.utils.utils  import quantization 
import torch 
import os
import torch
import argparse  

argparse = argparse.ArgumentParser(description='Test 3D CNN models on hurricane data') 
argparse.add_argument('--model', type=str, default='unet', help='Model to use for testing')
argparse.add_argument('--weight_path', type=str, default='/lcrc/project/ECP-EZ/jp/git/arcnn/checkpoints/hurricane_unet_no_penalty/best.pth', help='Path to the model weights')
argparse.add_argument('--rel_eb', type=float, default=0.005, help='Relative error bound for quantization') 
argparse.add_argument('--input_size', type=int, nargs=3, default=(100, 500, 500), help='Input size for the model') 
argparse.add_argument('--patch_size', type=int, default=50, help='Patch size for tiling prediction')
argparse.add_argument('--stride', type=int, default=45, help='Stride for tiling prediction')
argparse.add_argument('--batch_size', type=int, default=16, help='Batch size for prediction')
argparse.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on')


args = argparse.parse_args()
print(args)



def extract_blocks(volume, block_size, stride):
    blocks = []
    indices = []

    D, H, W = volume.shape
    d_steps = list(range(0, D - block_size + 1, stride))
    h_steps = list(range(0, H - block_size + 1, stride))
    w_steps = list(range(0, W - block_size + 1, stride))

    if d_steps[-1] != D - block_size:
        d_steps.append(D - block_size)
    if h_steps[-1] != H - block_size:
        h_steps.append(H - block_size)
    if w_steps[-1] != W - block_size:
        w_steps.append(W - block_size)

    for i in d_steps:
        for j in h_steps:
            for k in w_steps:
                block = volume[i:i+block_size, j:j+block_size, k:k+block_size]
                blocks.append(block)
                indices.append((i, j, k))
    print(f"Extracted {len(blocks)} blocks of size {block_size} with stride {stride}.")
    return np.stack(blocks), indices

def quantize_whole(data, abs_eb):
    precision = 1.0 / (2 * abs_eb)
    quantized, _ = quantization(data, precision)
    residuals = data - quantized
    residual_bound = np.max(np.abs(residuals))
    return quantized, residuals, residual_bound


def predict_blocks_torch(model, blocks, residual_bound, device='cuda', batch_size=16):
    model.eval()
    model.to(device)

    ## normalized to [-1, 1]
    normalized_blocks  = np.empty_like(blocks, dtype=np.float32)
    for i in range(blocks.shape[0]):
        block = blocks[i]
        min_val = block.min()
        max_val = block.max()
        normalized_blocks[i] = (block - min_val) / (max_val - min_val) * 2 - 1
    inputs = torch.from_numpy(normalized_blocks).unsqueeze(1).float().to(device)  # shape: [B, 1, D, H, W]
    preds = []
    with torch.no_grad():
        for i in range(0, inputs.shape[0], batch_size):
            batch = inputs[i:i+batch_size]
            out = model(batch).squeeze(1).cpu().numpy()
            out = out * residual_bound
            preds.append(out)
    return np.concatenate(preds)


def merge_blocks(indices, pred_blocks, quantized_blocks, full_shape, block_size):
    pred_vol = np.zeros(full_shape, dtype=np.float32)
    quant_vol = np.zeros(full_shape, dtype=np.float32)
    weight_map = np.zeros(full_shape, dtype=np.float32)

    for (i, j, k), pred_block, quant_block in zip(indices, pred_blocks, quantized_blocks):
        pred_vol[i:i+block_size, j:j+block_size, k:k+block_size] += quant_block + pred_block
        quant_vol[i:i+block_size, j:j+block_size, k:k+block_size] += quant_block
        weight_map[i:i+block_size, j:j+block_size, k:k+block_size] += 1

    pred_vol /= np.maximum(weight_map, 1e-6)
    quant_vol /= np.maximum(weight_map, 1e-6)
    return quant_vol, pred_vol

def tiling_pred_parallel_pytorch(data, decompressed,  block_size, stride, abs_eb, model, device='cuda', batch_size=16):
    
    # quantized, residuals, residual_bound = quantize_whole(data, abs_eb)
    
    quant_blocks, indices = extract_blocks(decompressed, block_size, stride)
    
    pred_blocks = predict_blocks_torch(model, quant_blocks, abs_eb, device, batch_size)
    quant_vol, pred_vol = merge_blocks(indices, pred_blocks, quant_blocks, data.shape, block_size)

    return  pred_vol


cur_device = args.device
cur_model = None
model_str = args.model.lower()
if(model_str == 'srcnn'):
    model = SRCNN()
elif(model_str == 'dncnn'):
    model = DnCNN3D()
elif(model_str == 'arcnn'):
    model = ARCNN()
elif(model_str == 'unet'):
    model = UNet3D() 
else:
    raise ValueError("Invalid model name. Choose from 'arcnn', 'srcnn', or 'dncnn'.") 
cur_model = model 

patch_size = args.patch_size
input_size = args.input_size
stride = args.stride
rel_eb = args.rel_eb 


# cur_model.to(cur_device)
model_path = args.weight_path 

cur_model.load_state_dict(torch.load(model_path, map_location=cur_device  ))
# cur_model.eval() 
time_step = 48
data = np.fromfile(f'/lcrc/project/SDR/pjiao/data/hurricane_all/clean/{time_step}/Pf{time_step:02}.bin.f32', 
                   dtype=np.float32).reshape(input_size)

print(0.012639424 * (data.max() - data.min()))

abs_eb = rel_eb * (data.max() - data.min()) 
print(f"Using absolute error bound: {abs_eb:.6f}") 

quantized, quant_vol, pred_vol = tiling_pred_parallel_pytorch(data, 
                                                    block_size=patch_size, 
                                                   stride=stride, 
                                                   abs_eb=abs_eb, 
                                                   model=cur_model, device=cur_device, batch_size=16)


orig_psnr = get_psnr(data, quant_vol)
print(f"Original PSNR: {orig_psnr:.4f}")
pred_psnr = get_psnr(data, pred_vol)
print(f"Predicted PSNR: {pred_psnr:.4f}")    
print(f"Quantized PSNR: {get_psnr(data, quantized):.4f}")    


