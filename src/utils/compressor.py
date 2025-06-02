import numpy as np 
import os
import subprocess
import uuid 

def edt_model(input_file, eb, shape, 
            edt_path = "/lcrc/project/ECP-EZ/jp/git/posterization_mitigation/build/test/test_quantize_and_edt"):
    # print("shape: ", shape) 
    shape_str = ' '.join(map(str, shape)) 
    random_num = np.random.randint(0, 1000000) 
    shape_len = len(shape) 
    command = f'{edt_path} -N {shape_len} -d {shape_str} \
                -m rel -e {eb} -i {input_file} \
                -q {random_num}.q -c {random_num}.c -t 16'  
    # os.system(command)
    result = subprocess.run(command,     stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    shell=True)
    # for line in result.stdout.split('\n'):
    #     if('PSNR' in line):
    #         print(line)
    #     if ('SSIM' in line):
    #         print(line)
    # result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
    c_data = np.fromfile(f'{random_num}.c', dtype=np.float32).reshape(shape) 
    os.remove(f'{random_num}.c')
    os.remove(f'{random_num}.q') 
    return c_data 


def sz3(input_data, eb, shape, config = None, 
        compressor_path = "/lcrc/project/ECP-EZ/jp/git/SZ3/build/tools/sz3/sz3" ):
    # random_num = np.random.randint(0, 1000000)
    random_num = uuid.uuid4()
    input_file  = f"/tmp/input_file{random_num}.dat" 
    input_data.astype(np.float32).tofile(input_file) 
    sz3_shape = shape[::-1]
    shape_str = ' '.join(map(str, sz3_shape)) 
    N = len(shape) 
    sz3_path= compressor_path
    zfile = f"/tmp/{random_num}.sz3" 
    ofile = f"/tmp/{random_num}.sz3.out" 
    if config is not None:
        command = f"{sz3_path} -f -i {input_file} -z {zfile} -o {ofile}  -M REL {eb} -{N} {shape_str} -c {config}" 
    else: 
        command = f"{sz3_path} -f -i {input_file} -z {zfile} -o {ofile}  -M REL {eb} -{N} {shape_str}"
    # print(command) 
    # os.system(command)
    result = subprocess.run(command,     stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    shell=True)
    output = np.fromfile(ofile, dtype=np.float32).reshape(shape)
    cr = os.path.getsize(input_file) / os.path.getsize(zfile) 
    os.remove(zfile)
    os.remove(ofile)
    os.remove(input_file) 
    return output ,cr 

def cusz(input_data, eb, shape, config = None, 
        compressor_path = "/lcrc/project/ECP-EZ/jp/git/cusz_stable/build/cusz" ):
    random_num = uuid.uuid4()
    input_file  = f"/tmp/input_file{random_num}.dat" 
    input_data.astype(np.float32).tofile(input_file) 
    sz3_shape = shape[::-1]
    shape_str = '-'.join(map(str, sz3_shape))
    compress_command = f"{compressor_path} -t f32 -m rel -e {eb} -i {input_file} -l {shape_str} -z"
    decompress_command = f"{compressor_path} -i {input_file}.cusza -x "
    # print(compress_command)
    # print(decompress_command) 
    command = f"{compress_command}  && {decompress_command}"  
    result = subprocess.run(command, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    shell=True)
    output_file = f"{input_file}.cuszx"
    output = np.fromfile(output_file, dtype=np.float32).reshape(shape)
    cr = os.path.getsize(input_file) / os.path.getsize(f"{input_file}.cusza")
    os.remove(input_file)
    os.remove(f"{input_file}.cusza")
    os.remove(output_file)
    return output, cr



    
    
    