import numpy as np 
import ctypes
from skimage.util import view_as_windows

def get_psnr(src_data, dec_data):

    data_range = np.max(src_data) - np.min(src_data)
    diff = src_data - dec_data
    max_diff = np.max(abs(diff))
    print("abs err={:.8G}".format(max_diff))
    mse = np.mean(diff ** 2)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr 

def compute_patch_psnr_batch(src_patches, dec_patches):
    """
    Computes PSNR between each pair of patches in two arrays.
    Both inputs should be [N, H, W].
    Returns: [N] array of PSNR values.
    """
    assert src_patches.shape == dec_patches.shape
    N = src_patches.shape[0]

    src_flat = src_patches.reshape(N, -1)
    dec_flat = dec_patches.reshape(N, -1)

    data_range = src_flat.max(axis=1) - src_flat.min(axis=1)
    diff = src_flat - dec_flat
    mse = np.mean(diff ** 2, axis=1)
    mse = np.maximum(mse, 1e-10)  # avoid log(0)

    psnr = 20 * np.log10(data_range + 1e-8) - 10 * np.log10(mse)
    return psnr

def qcatssim(orig:np.ndarray[np.float32], decompressed:np.ndarray[np.float32]):
    dims = np.array(orig.shape)[::-1]
    # print(dims)
    lib = ctypes.CDLL("/lcrc/project/ECP-EZ/jp/git/posterization_mitigation/build/test/liblibqcatssim.so")
    lib.calculateSSIM.restype = ctypes.c_double
    lib.calculateSSIM.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32),
                              np.ctypeslib.ndpointer(dtype=np.float32),
                              np.ctypeslib.ndpointer(dtype=np.int32),
                              ctypes.c_int]
    result =lib.calculateSSIM(
        orig, 
        decompressed,
        dims.astype(np.int32), 
        ctypes.c_int(dims.size))
    return result 

def get_ssim(src_data, dec_data):
    k1 = 0.01
    k2 = 0.03 
    winsize, winshift = 7, 2
    # sub windows of the original and decompressed images
    ndim = len(src_data.shape)
    window_shape = (winsize,) * ndim
    src_data_windows  = view_as_windows(src_data, window_shape , step=winshift)
    dec_data_windows = view_as_windows(dec_data, window_shape , step=winshift) 
    # for each sub window, calculate the mean and variance 
    num_windows = src_data_windows.shape[0] 
    
    return None 

