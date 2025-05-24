import os 
import subprocess    
import argparse 
import sys 
from concurrent.futures import ThreadPoolExecutor as executor  
import re

def time_step_filter(input_files, start_time_step, end_time_step):   
    """ time step filter
    includesive on both end
    """ 
    filtered_files = []
    for file in input_files:
        # find the number of time steps in the file name 
        # assume the file name is Uf02.bin.f32 
        # so 02 will be the time step 
        # remove the .bin.f32 first
        file_name = os.path.basename(file)
        file_name = file_name.split('.')[0]
        pattern = r'(\d+)'  # regex pattern to match digits 
        match = re.search(pattern, file_name) 
        if match:
            time_step = int(match.group(1)) 
            if time_step >= start_time_step and time_step <= end_time_step:
                filtered_files.append(file) 

    return filtered_files 
    


parser = argparse.ArgumentParser(
                    prog='prepare_data sets',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-i', '--inputdir', type=str, required=True, help='input data file') 
parser.add_argument('-o', '--outputdir', type=str, required=True, help='output data directory')
parser.add_argument('-d', '--dtype', type=str, default='f32', help='data type')
parser.add_argument('-n', '--num', type=int, default=1, help='dimension of data') 
parser.add_argument('-s', '--shape', nargs='+', type=int, default=1, help='shape of data')
parser.add_argument("--artifact", type=str, default='banding', help='artifacts type') 
parser.add_argument('--ext', type=str, default='.f32', help='extension of the data files') 
parser.add_argument('--use', type=str, default='train', help='prefix of the data folder') 

args = parser.parse_args()

artifact = args.artifact     
# get a list of all the files in the input directory  
input_files = [f for f in os.listdir(args.inputdir) if f.endswith(args.ext)]
print(input_files) 



fildterd_files = input_files
print(fildterd_files) 

# exit(0) 
def process_file(inputfile, rel_eb):
    # Construct the full input file path
    # command = "pwd && ls -l"
    
    command = f" {sys.executable} ./prepare_{artifact}.py  \
                -i {inputfile}  -o {args.outputdir} \
                -e {rel_eb}  -d {args.dtype} \
                -n {args.num} \
                -s {' '.join(map(str, args.shape))} \
                --artifact {artifact} \
                --use {args.use} "
    print(command) 
    subprocess.run(command, shell=True, check=True)   

# Loop through each file and process it
# for inputfile in fildterd_files:
#     inputfile_path = os.path.join(args.inputdir, inputfile)
#     rel_eb = 0.001 
#     process_file(inputfile_path, rel_eb)
    
# parallel processing 
rel_eb = 0.001 
input_file_paths = [os.path.join(args.inputdir, inputfile) for inputfile in fildterd_files] 
with executor(max_workers=16) as ex:
    for inputfile in input_file_paths:
        ex.submit(process_file, inputfile, rel_eb)
