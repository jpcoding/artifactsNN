#!/bin/bash -l
#PBS -N gpu-test
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -l walltime=20:00:00
#PBS -A SDR
#PBS -M pujiao@uky.edu
#PBS -m bea 

cd $PBS_O_WORKDIR
echo Working directory is $PBS_O_WORKDIR

source /lcrc/project/SDR/pjiao/apps/miniconda/bin/activate
conda activate py310 

python train_residual_3d_mem_chunks.py  \
 --train_input /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/train/ \
 --train_target  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/train/ \
 --val_input /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/val/ \
 --val_target  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/val/ \
  --shape 100 500 500  --dtype f32 \
  --train yes  \
  --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/train/dncnn_U_no_penalty/  \
  --model dncnn --batch_size 8  --bestpath  /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/hurricane_dncnn_U_1_10_no_penalty.pth\
  --range_penalty 0 --num_epoches 100 


python train_residual_3d_mem_chunks.py  \
 --train_input /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/train/ \
 --train_target  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/train/ \
 --val_input /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/val/ \
 --val_target  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/val/ \
  --shape 100 500 500  --dtype f32 \
  --train yes  \
  --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/train/unet_U_no_penalty/  \
  --model dncnn --batch_size 8  --bestpath  /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/hurricane_unet_U_1_10_no_penalty.pth\
  --range_penalty 0 --num_epoches 100 



# python train_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/nyx_velocity_x.f32/decompressed/ \
#  --target_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/nyx_velocity_x.f32/orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/nyx_velocity_x.f32/model_radial_mimo \
#    --model dncnn  --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_nyx_radial_dncnn.pth \
#    --num_train 2000 

# python train_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/decompressed/ \
#  --target_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/model_radial_mimo \
#    --model mimounet  --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_hurricane_radial_mimo.pth \
#    --num_train 2000 

# python train_arcnn_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/decompressed/ \
#  --target_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/model_radial_no \
#    --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_hurricane_radial_no_panealty.pth \
#    --range_penalty 0 --num_train 2000 --penalty_fn sign_penalty  --penalty_order 2.0 



# python train_arcnn_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/decompressed/ \
#  --target_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/model_radial_sign_0.1 \
#    --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_hurricane_radial_sign_0.1.pth \
#    --range_penalty 0.1 --num_train 2000 --penalty_fn sign_penalty  --penalty_order 2.0 

# python train_arcnn_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/decompressed/ \
#  --target_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/hurricane_Vf48.bin.f32/model_radial_range_0.1 \
#    --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_hurricane_radial_range_0.1.pth \
#    --range_penalty 0.1 --num_train 2000 --penalty_fn range_penalty  --penalty_order 2.0 

# python train_arcnn_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/NYX_velocity_x.f32/quantized_data/ \
#  --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/NYX_velocity_x.f32/orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/NYX_velocity_x.f32/model_updated_sign_2/ \
#    --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_nyx_p_sign_2.pth \
#    --range_penalty 0.1 --num_train 2000 --penalty_fn sign_penalty  --penalty_order 2.0 

# python train_arcnn_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/NYX_velocity_x.f32/quantized_data/ \
#  --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/NYX_velocity_x.f32/orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/NYX_velocity_x.f32/model_updated_sign_1/ \
#    --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_nyx_p_sign_1.pth \
#    --range_penalty 0.1 --num_train 2000 --penalty_fn sign_penalty  --penalty_order 1.0 