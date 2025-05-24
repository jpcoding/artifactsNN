 python data_stats.py -i /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/orig/ \
  -o   /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane/decompressed/banding/0.001/ \
    -d f32 -n 3 -s 100 500 500  --ext f32

python prepare_dataset.py -i  /lcrc/project/SDR/pjiao/data/hurricane_all/clean/ \
     -o /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane \
     -d f32 -n 3 -s 100 500 500 --artifact banding --ext f32 \
     --use train 