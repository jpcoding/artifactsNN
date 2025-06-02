import numpy as np 
import ctypes
from skimage.util import view_as_windows
from numba import njit, prange

def get_psnr(src_data, dec_data):

    data_range = np.max(src_data) - np.min(src_data)
    diff = src_data - dec_data
    max_diff = np.max(abs(diff))
    # print("abs err={:.8G}".format(max_diff))
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
    return qcatssim(src_data, dec_data) 

# def get_ssim(src_data, dec_data):
#     k1 = 0.01
#     k2 = 0.03 
#     winsize, winshift = 7, 2
#     # sub windows of the original and decompressed images
#     ndim = len(src_data.shape)
#     window_shape = (winsize,) * ndim
#     src_data_windows  = view_as_windows(src_data, window_shape , step=winshift)
#     dec_data_windows = view_as_windows(dec_data, window_shape , step=winshift) 
#     # for each sub window, calculate the mean and variance 
#     num_windows = src_data_windows.shape[0] 
    
#     return None 


# @njit
# def SSIM_3d_window(orig, other):
#     K1 = 0.01
#     K2 = 0.03
#     xMin = orig.min()
#     xMax = orig.max()
#     yMin = other.min()
#     yMax = other.max()
#     xMean = orig.mean()
#     yMean = other.mean()
#     var_x = orig.var()
#     var_y = other.var()
#     var_xy = ((orig - xMean) * (other - yMean)).mean()
#     xSigma = np.sqrt(var_x)
#     ySigma = np.sqrt(var_y)
#     if (xMax - xMin) == 0:
#         c1 = K1**2
#         c2 = K2**2
#     else:
#         L = xMax - xMin
#         c1 = (K1 * L) ** 2
#         c2 = (K2 * L) ** 2

#     c3 = c2 / 2
#     luminance = (2 * xMean * yMean + c1) / (xMean**2 + yMean**2 + c1)
#     contrast  = (2 * xSigma * ySigma + c2) / (xSigma**2 + ySigma**2 + c2)
#     structure = (var_xy + c3) / (xSigma * ySigma + c3)
#     ssim = luminance * contrast * structure
#     return ssim

# @njit(parallel=True)
# def SSIM_3d_parallel(orig_windows, other_windows):
#     z, y, x = orig_windows.shape[:3]
#     total = 0.0
#     print("orig_windows.shape", (z * y * x))
#     for i in prange(z):
#         for j in range(y):
#             for k in range(x):
#                 total += SSIM_3d_window(orig_windows[i, j, k], other_windows[i, j, k])

#     return total / (z * y * x)

# def SSIM_3d(orig, other, window_shape=(7, 7, 7), stride=2):
#     orig_windows = view_as_windows(orig, window_shape, step=stride)
#     other_windows = view_as_windows(other, window_shape, step=stride)
#     return SSIM_3d_parallel(orig_windows, other_windows)
