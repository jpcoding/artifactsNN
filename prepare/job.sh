#!/bin/bash -l
#PBS -N tets_unet
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -l walltime=10:00:00
#PBS -A SDR
#PBS -m bea 

cd $PBS_O_WORKDIR
echo Working directory is $PBS_O_WORKDIR

source /lcrc/project/SDR/pjiao/apps/miniconda/bin/activate
conda activate py312

python prepare_zarr_2.py

