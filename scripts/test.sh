# passed
# python ../test/train_residual_3d_mem_chunks.py  \
#  --train_input /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/train/ \
#  --train_target  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/train/ \
#  --val_input /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/val/ \
#  --val_target  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/val/ \
#   --shape 100 500 500  --dtype f32 \
#   --train yes  \
#   --outputs_dir /tmp/dncnn_no/  \
#   --model dncnn --batch_size 8  --bestpath  /tmp/test.pth \
#   --range_penalty 0 --num_epoches 1

# passed 
python ../test/train_residual_3d.py  \
 --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_block/miranda_velocityx.f32/decompressed/ \
 --target_dir  /lcrc/project/SDR/pjiao/data/cnn_train/vx3d_block/miranda_velocityx.f32/orig_data/ \
  --shape 64 64 64 --dtype f32 \
  --entropy none --train yes  \
  --outputs_dir /tmp/no \
  --model dncnn --batch_size 8  --bestpath /tmp/best.pth \
  --range_penalty 0  --num_train 10 --penalty_fn range_penalty  --penalty_order 2.0  --num_epoches 1 