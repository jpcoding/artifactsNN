#!/bin/bash -l
#PBS -N tets_unet
#PBS -l select=1:ngpus=1
#PBS -j oe
#PBS -l walltime=10:00:00
#PBS -A SDR
#PBS -m bea 
#PBS -M pujiao@uky.edu



cd $PBS_O_WORKDIR
echo Working directory is $PBS_O_WORKDIR

source /lcrc/project/SDR/pjiao/apps/miniconda/bin/activate
conda activate py312

python ../test/train_residual_3d_zarr.py --data_dir x --shape 64 64 64\
 --dtype f32 --entropy None --train yes --outputs_dir /lcrc/project/ECP-EZ/jp/git/arcnn/checkpoints/hurricane_unet_no_penalty_ssz3i/ --model unet \
  --batch_size 8 \
  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/checkpoints/hurricane_unet_no_penalty_ssz3i/best.pth \
  --num_epoches 100 --compressor sz3i --eb 5e-03 


# python ../test/train_residual_3d_zarr.py --data_dir x --shape 64 64 64\
#  --dtype f32 --entropy None --train yes --outputs_dir /lcrc/project/ECP-EZ/jp/git/arcnn/checkpoints/hurricane_unet_no_penalty/ --model unet \
#   --batch_size 8 \
#   --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/checkpoints/hurricane_unet_no_penalty/best.pth \
#   --num_epoches 100 

# python train_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32/quantized_data/ \
#  --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32/model_unet_no_penalty/ \
#    --model unet --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_miranda_unet_no_penalty.pth \
#    --range_penalty 0 --num_train 1000 --penalty_fn range_penalty  --penalty_order 2.0 

# python train_arcnn_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//quantized_data/ \
#  --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//model_updated_range_2_0.5/ \
#    --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_miranda_range_2_0.5.pth \
#    --range_penalty 0.5 --num_train 1000 --penalty_fn range_penalty  --penalty_order 2.0 


# python train_arcnn_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//quantized_data/ \
#  --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//model_updated_range_2_1.0/ \
#    --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_miranda_range_2_1.0.pth \
#    --range_penalty 1.0  --num_train 1000 --penalty_fn range_penalty  --penalty_order 2.0 


# python train_arcnn_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//quantized_data/ \
#  --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//model_updated_sign_2/ \
#    --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_miranda_sign_2.pth \
#    --range_penalty 0.5 --num_train 1000 --penalty_fn sign_penalty  --penalty_order 2.0 


# python train_arcnn_residual_3d.py  \
#  --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//quantized_data/ \
#  --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//orig_data/ \
#   --shape 64 64 64 --dtype f32 \
#   --entropy none --train yes  \
#    --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32//model_updated_sign_1/ \
#    --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_miranda_sign_1.pth \
#    --range_penalty 0.5 --num_train 1000 --penalty_fn sign_penalty  --penalty_order 1.0 