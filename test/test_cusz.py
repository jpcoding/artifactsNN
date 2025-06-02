import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) 
from src.utils.compressor import sz3, cusz 
from src.utils.stats import get_psnr 
import uuid 
import numpy as np


data = np.fromfile('/home/pjiao/data/SDRBENCH-Miranda-256x384x384/velocityx.f32', 
                   dtype=np.float32).reshape(256, 384, 384)
compressed_data, compression_ratio = cusz(data, eb=1e-3, shape=data.shape)
print(f"Compression Ratio: {compression_ratio:.2f}")
print(f"psnr: {get_psnr(data, compressed_data):.2f}")
