from test.data_prepare import quantization 
import sys 
import numpy as np 


data  = np.fromfile(sys.argv[1], dtype=np.float32) 
eb = float(sys.argv[2]) 
abs_eb = eb*(np.max(data) - np.min(data))   
recip_precision = 1.0 /(2* abs_eb) 
quantized_data, quant_idx = quantization(data, recip_precision)
quantized_data.tofile(sys.argv[3], format='%f')
quant_idx.tofile(sys.argv[4])
