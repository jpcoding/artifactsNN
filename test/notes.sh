python train_arcnn_residual.py  \
 --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/patches_64_32_orig/input/ \
 --target_dir /lcrc/project/SDR/pjiao/data/cnn_train/patches_64_32_orig/target/ \
 --shape 1 64 64 --dtype f32 --entropy none --train yes  \
 --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/patches_64_32_orig/models/ \
 --model dncnn --batch_size 128

python train_arcnn_residual_3d.py  \
 --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32/quantized_data/ \
 --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32/orig_data/ \
  --shape 64 64 64 --dtype f32 \
  --entropy none --train yes  \
   --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32/model_updated/ \
   --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_dncnn_64_64_64_f32_best.pth \
   --range_penalty 0.1 

 python data_prepare_3d.py  -i $DATA/SDRBENCH-Miranda-256x384x384/velocityx.f32 \
  -o /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/ -e 0.001 -d f32  -n 3 -s 256 384 384

python data_prepare_3d.py  -i $DATA/NYX512x512x512/velocity_y.f32 \
  -o /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/ -e 0.01 -d f32  -n 3 -s 512 512 512 


python data_prepare_3d.py  -i $DATA/NYX512x512x512/velocity_x.f32 \
  -o /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/ -e 0.05 -d f32  -n 3 -s 512 512 512 \
  --prefix NYX

python train_arcnn_residual_3d.py  \
 --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/NYX_velocity_y.f32/quantized_data/ \
 --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32/orig_data/ \
  --shape 64 64 64 --dtype f32 \
  --entropy none --train yes  \
   --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32/model_updated/ \
   --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_nyx_p2.pth \
   --range_penalty 0.1 


 python data_prepare_3d_block.py  -i $DATA/SDRBENCH-Miranda-256x384x384/velocityx.f32 \
  -o /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_block/ -e 0.01 -d f32  -n 3 -s 256 384 384


python train_arcnn_residual_3d.py  \
 --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/NYX_velocity_y.f32/quantized_data/ \
 --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32/orig_data/ \
  --shape 64 64 64 --dtype f32 \
  --entropy none --train yes  \
   --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/velocityx.f32/model_updated/ \
   --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_nyx_p2.pth \
   --range_penalty 0.1 

python train_arcnn_residual_3d.py  \
 --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_block/miranda_velocityx.f32/decompressed/ \
 --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_block/miranda_velocityx.f32/orig_data/ \
  --shape 64 64 64 --dtype f32 \
  --entropy none --train yes  \
  --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_block/miranda_velocityx.f32/model_no_penalty \
  --model dncnn --batch_size 8  --bestpath /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/vx3d_miranda_block_no_penalty.pth \
  --range_penalty 0  --num_train 1000 --penalty_fn range_penalty  --penalty_order 2.0 

 python data_prepare_3d_block.py  -i /home/pjiao/data/hurricane100x500x500/Vf48.bin.f32 \
  -o /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_block/ -e 0.01 -d f32  -n 3 -s 100 500 500 --prefix hurricane


python data_prepare_3d_radial.py  -i /home/pjiao/data/hurricane100x500x500/Vf48.bin.f32  \
 -o /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/ -e 0.01 -d f32  \
 -n 3 -s 100 500 500 --prefix hurricane


python data_prepare_3d_radial.py  -i "/home/pjiao/data/NYX512x512x512/velocity_x.f32"  \
 -o /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_radial/ -e 0.01 -d f32  \
 -n 3 -s 512 512 512 --prefix nyx

python train_residual_3d_mem_chunks.py  \
 --train_input /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/train/ \
 --train_target  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/train/ \
 --val_input /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/val/ \
 --val_target  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/val/ \
  --shape 100 500 500  --dtype f32 \
  --entropy none --train no  \
  --outputs_dir ./  \
  --model dncnn --batch_size 8  --bestpath none \
  --range_penalty 0 

python train_residual_3d_mem_chunks.py  \
 --train_input /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/train/ \
 --train_target  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/train/ \
 --val_input /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/val/ \
 --val_target  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/val/ \
  --shape 100 500 500  --dtype f32 \
  --train yes  \
  --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/train/dncnn_no/  \
  --model dncnn --batch_size 8  --bestpath  /lcrc/project/ECP-EZ/jp/git/arcnn/test/weights/hurricane_t_1_10_no_penalty.pth\
  --range_penalty 0 --num_epoches 1