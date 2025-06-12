
## residual model 
import numpy as np 
from matplotlib import pyplot as plt
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('./'), '../'))) 

from src.models import  SRCNN, ARCNN, DnCNN3D, UNet3D
from src.training import train_model, test_model 
from src.utils.stats import get_psnr, qcatssim
import torch 
import os
os.environ["CUDNN_V8_API_ENABLED"] = "1"
import torch
import torch.nn as nn
from baseline import edt_model 
import argparse  


argparse = argparse.ArgumentParser(description='Test 3D CNN models on hurricane data') 
argparse.add_argument('--model', type=str, default='unet', help='Model to use for testing')
argparse.add_argument('--weight_path', type=str, default='/lcrc/project/ECP-EZ/jp/git/arcnn/checkpoints/hurricane_unet_no_penalty/best.pth', help='Path to the model weights')
argparse.add_argument('--rel_eb', type=float, default=0.005, help='Relative error bound for quantization') 
argparse.add_argument('--input_size', type=int, nargs=3, default=(100, 500, 500), help='Input size for the model') 
argparse.add_argument('--patch_size', type=int, default=50, help='Patch size for tiling prediction')
argparse.add_argument('--stride', type=int, default=45, help='Stride for tiling prediction')

args = argparse.parse_args()
print(args)

def calculate_total_memory(model, input_size):
    # Get model parameters memory
    total_params = sum(p.numel() for p in model.parameters())
    param_memory = total_params * 4 / (1024 ** 2)  # Convert to MB
    
    # Get the number of elements in the input tensor
    input_tensor = torch.randn(1, *input_size)  # Batch size of 1
    activation_memory = 0
    
    # Forward pass to compute activations and add their memory
    with torch.no_grad():
        x = input_tensor
        for layer in model.dncnn:
            x = layer(x)
            # Calculate memory used by the activations
            activation_memory += x.numel() * 4 / (1024 ** 2)  # Convert to MB
    
    # Total memory is the sum of parameters and activations
    total_memory = param_memory + activation_memory
    return total_memory, param_memory, activation_memory


def quantization(data: np.ndarray, recip_precision: float):
    quant_idx = np.round(data * recip_precision).astype(np.int32)    
    quantized_data = (quant_idx / recip_precision).astype(data.dtype) 
    return quantized_data, quant_idx.astype(np.int32) 

def report(model_str, modelpath, original_data,  abs_eb, normailze_pred = False, 
           lazy = False, learning_residual = True): 
    orig = original_data
    precition = 1.0 /(2* abs_eb)  
    quantized, _  = quantization(orig, precition) 
    residuals = orig - quantized
    residual_bound = abs_eb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    if(model_str == 'srcnn'):
        model = SRCNN().to(device)
    elif(model_str == 'dncnn'):
        model = DnCNN3D().to(device)
    elif(model_str == 'arcnn'):
        model = ARCNN().to(device)
    elif(model_str == 'unet'):
        model = UNet3D().to(device) 
    else:
        raise ValueError("Invalid model name. Choose from 'arcnn', 'srcnn', or 'dncnn'.") 
    
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.eval()
    quantized_copy = quantized.copy() 
    quantized = (quantized - quantized.min()) / (quantized.max() - quantized.min())*2-1
    # print("quantized max: ", quantized.max(), "quantized min: ", quantized.min()) 
    input_max = quantized.max() 
    input_min = quantized.min() 
    input_tensor = torch.from_numpy(quantized).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_residual = model(input_tensor)
        pred_residual = pred_residual.squeeze().cpu().numpy()

    if learning_residual:
        if normailze_pred == True : pred_residual = pred_residual/(np.max(np.abs(pred_residual)))
        pred_residual = pred_residual * residual_bound
        pred = quantized_copy + pred_residual
    else:
        pred = pred_residual 
        pred = (pred+1)*0.5 * (quantized_copy.max() - quantized_copy.min()) + quantized_copy.min() 
    quantized = quantized_copy
    if lazy == False : 
        orig_psnr = get_psnr(orig, quantized_copy)
        orig_ssim = qcatssim(orig, quantized_copy) 
        pred_psnr = get_psnr(orig, pred)
        pred_ssim = qcatssim(orig, pred)
        
        # print("orig psnr: ", orig_psnr)
        # print("orig ssim: ", orig_ssim)
        # print("pred psnr: ", pred_psnr)
        # print("pred ssim: ", pred_ssim)
    else:
        orig_psnr = 0
        orig_ssim = 0
        pred_psnr = 0
        pred_ssim = 0
    torch.cuda.empty_cache()
    return quantized, pred, orig_psnr, orig_ssim, pred_psnr, pred_ssim

def plot(orig, quantized, pred, edt_result, orig_psnr, 
         orig_ssim, pred_psnr, pred_ssim, edt_psnr, edt_ssim, 
          plot_location ):    
    # plot the results 
    figs1,ax1 = plt.subplots(1, 4, figsize=(16, 4))

    slices = plot_location 
    v_min = np.min(orig[slices]) 
    v_max = np.max(orig[slices]) 

    ax1[0].imshow(orig[slices], cmap='coolwarm')
    ax1[0].set_title('original')
    ax1[1].imshow(quantized[slices], cmap='coolwarm', vmin=v_min, vmax=v_max)
    ax1[1].set_title(f'Quantized, PSNR: {orig_psnr:.2f}, SSIM: {orig_ssim:.2f}')
    ax1[2].imshow(pred[slices], cmap='coolwarm', vmin=v_min, vmax=v_max)
    ax1[2].set_title(f'Predicted, PSNR: {pred_psnr:.2f}, SSIM: {pred_ssim:.2f}')
    ax1[3].imshow(edt_result[slices], cmap='coolwarm', vmin=v_min, vmax=v_max)
    ax1[3].set_title(f'EDT , PSNR: {edt_psnr:.2f}, SSIM: {edt_ssim:.2f}')
    # plot error distribution 1
    error_quantized = orig - quantized
    error_pred = orig - pred
    error_residual = pred-quantized
    edt_error = orig - edt_result 
    edt_compensated = edt_result - quantized 
    figs2, ax = plt.subplots(1, 5, figsize=(15,5))
    e_max = np.max(abs(error_quantized[slices]))
    e_min = -e_max
    ax[0].imshow(error_quantized[slices], cmap='coolwarm', vmin=e_min, vmax=e_max)
    ax[0].set_title('Original residual')
    ax[1].imshow(error_pred[slices], cmap='coolwarm', vmin=e_min, vmax=e_max)
    ax[1].set_title('Post pred error') 
    ax[2].imshow(error_residual[slices], cmap='coolwarm', vmin=e_min, vmax=e_max)
    ax[2].set_title('Learned residual')
    ax[3].imshow(edt_error[slices], cmap='coolwarm', vmin=e_min, vmax=e_max)    
    ax[3].set_title('Post EDT error')
    ax[4].imshow(edt_compensated[slices], cmap='coolwarm', vmin=e_min, vmax=e_max)    
    ax[4].set_title('EDT compensation')
    
# def tiling_pred(data, block_size, stride, global_abs_eb, model, weight_path):
#     shape = data.shape
#     pred_global_overlap = np.zeros_like(data, dtype=np.float32)
#     quantized_global_overlap = np.zeros_like(data, dtype=np.float32)
#     weight_overlap = np.zeros_like(data, dtype=np.float32) 
#     num_blocks = 0 
#     for i in range(0, shape[0] - block_size + 1, stride):
#         for j in range(0, shape[1] - block_size + 1, stride):
#             for k in range(0, shape[2] - block_size + 1, stride):
#                 x_small = data[i:i+block_size, j:j+block_size, k:k+block_size].copy()
#                 quantized, pred, orig_psnr, orig_ssim, pred_psnr, pred_ssim  = report(model,
#                         weight_path,
#                     x_small, global_abs_eb, normailze_pred=False,
#                     learning_residual=True)
#                 pred_global_overlap[i:i+block_size, j:j+block_size, k:k+block_size] += pred
#                 quantized_global_overlap[i:i+block_size, j:j+block_size, k:k+block_size] += quantized
#                 weight_overlap[i:i+block_size, j:j+block_size, k:k+block_size] += 1
#                 num_blocks += 1 
#     quantized_global_overlap /= np.maximum(weight_overlap, 1e-6)
#     pred_global_overlap /= np.maximum(weight_overlap, 1e-6)
#     print("num blocks: ", num_blocks)    
#     return quantized_global_overlap, pred_global_overlap 

def tiling_pred(data, block_size, stride, global_abs_eb, model, weight_path):
    shape = data.shape
    pred_global_overlap = np.zeros_like(data, dtype=np.float32)
    quantized_global_overlap = np.zeros_like(data, dtype=np.float32)
    weight_overlap = np.zeros_like(data, dtype=np.float32)
    num_blocks = 0

    # Compute tile positions including trailing edges
    x_positions = list(range(0, shape[0] - block_size + 1, stride))
    y_positions = list(range(0, shape[1] - block_size + 1, stride))
    z_positions = list(range(0, shape[2] - block_size + 1, stride))

    if x_positions[-1] != shape[0] - block_size:
        x_positions.append(shape[0] - block_size)
    if y_positions[-1] != shape[1] - block_size:
        y_positions.append(shape[1] - block_size)
    if z_positions[-1] != shape[2] - block_size:
        z_positions.append(shape[2] - block_size)

    for i in x_positions:
        for j in y_positions:
            for k in z_positions:
                x_small = data[i:i+block_size, j:j+block_size, k:k+block_size].copy()
                quantized, pred, orig_psnr, orig_ssim, pred_psnr, pred_ssim = report(
                    model, weight_path, x_small,
                    global_abs_eb, normailze_pred=False,
                    learning_residual=True
                )
                pred_global_overlap[i:i+block_size, j:j+block_size, k:k+block_size] += pred
                quantized_global_overlap[i:i+block_size, j:j+block_size, k:k+block_size] += quantized
                weight_overlap[i:i+block_size, j:j+block_size, k:k+block_size] += 1
                num_blocks += 1

    quantized_global_overlap /= np.maximum(weight_overlap, 1e-6)
    pred_global_overlap /= np.maximum(weight_overlap, 1e-6)

    print("num blocks:", num_blocks)
    return quantized_global_overlap, pred_global_overlap


from baseline import edt_model 


orig_psnr_list = []
orig_ssim_list= []
pred_psnr_list = []
pred_ssim_list = []
edt_ssim_list = [] 
edt_psnr_list = []
max_error_edt_list = []
max_error_pred_list = []
max_error_orig_list = []  


model = args.model 
weight_path = args.weight_path 
input_size = args.input_size
patch_size = args.patch_size
patch_stride = args.stride 
print(f"Testing model: {args.model},\
     weight path: {weight_path},\
         input size: {input_size},\
             patch size: {patch_size},\
                 patch stride: {patch_stride}") 

for i in range(1,2):
    time_step = i
    path = f'/lcrc/project/SDR/pjiao/data/hurricane_all/clean/{time_step}/Pf{time_step:02}.bin.f32'
    # path =  "/home/pjiao/data/hurricane100x500x500/TCf48.bin.f32"
    data = np.fromfile(path, dtype=np.float32).reshape(input_size)
    # plt.imshow(data, cmap='coolwarm')
    # x = data[0:1800, 0:1800].copy()
    # x = data[0:100, 0:100, 0:100].copy()
    # x = data
    # x = data[0:100, 128:128+128, 128:129+128].copy()
    x = data
    rel_eb = args.rel_eb
    abs_eb = rel_eb * (x.max() - x.min()) 
    
    # weight_path = '/lcrc/project/ECP-EZ/jp/git/arcnn/test_bak/weights/vx3d_miranda_unet_no_penalty_orig.pth' 
    # weight_path = '/lcrc/project/ECP-EZ/jp/git/arcnn/test_bak/weights/hurricane_dncnn_U_1_10_no_penalty.pth'
    # weight_path = '/lcrc/project/ECP-EZ/jp/git/arcnn/test_bak/weights/hurricane_unet_U_1_10_no_penalty.pth'
    # weight_path = '/lcrc/project/ECP-EZ/jp/git/arcnn/checkpoints/hurricane_dncnn_no_penalty/best.pth'
    weight_path = args.weight_path 
    # quantized, pred, orig_psnr, orig_ssim, pred_psnr, pred_ssim  = report('unet',
    #         weight_path,
    #     x, rel_eb, normailze_pred=False,lazy= True,
    #     learning_residual=True)
    quantized, pred = tiling_pred(x, patch_size, patch_stride, abs_eb, model , weight_path) 
    orig_psnr = get_psnr(x, quantized)
    orig_ssim = qcatssim(x, quantized)
    pred_psnr = get_psnr(x, pred)
    pred_ssim = qcatssim(x, pred)
    max_error_pred = np.max(np.abs(pred - x))/(x.max() - x.min())
    max_error_orig = np.max(np.abs(quantized - x))/(x.max() - x.min())  
    orig_psnr_list.append(orig_psnr)    
    orig_ssim_list.append(orig_ssim)
    pred_psnr_list.append(pred_psnr)
    pred_ssim_list.append(pred_ssim)
    max_error_pred_list.append(max_error_pred)
    max_error_orig_list.append(max_error_orig)
    print(f"Time step {i}: orig psnr: {orig_psnr}, pred psnr: {pred_psnr}, pred ssim: {pred_ssim}") 
    
# exit()
# import pandas as pd  
# pd.DataFrame({
#     'orig_psnr': orig_psnr_list,
#     'pred_psnr': pred_psnr_list,
#     'orig_ssim': orig_ssim_list,
#     'pred_ssim': pred_ssim_list,
#     'max_error_orig': max_error_orig_list,
#     'max_error_pred': max_error_pred_list,
# }).to_csv(f'hurricane_{model}_{rel_eb}_{patch_size}_{patch_stride}.csv')
    
