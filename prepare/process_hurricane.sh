#!/bin/bash

root_dir="/lcrc/project/SDR/pjiao/data/hurricane_all/clean/"

for i in $(seq 1 8);
do 
    echo "Processing hurricane $i"
    python prepare_dataset.py -i ${root_dir}/${i} \
         -o /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane \
         -d f32 -n 3 -s 100 500 500 --artifact banding --ext f32 --use train 
done

for i in $(seq 9 10);
do 
    echo "Processing hurricane $i"
    python prepare_dataset.py -i ${root_dir}/${i} \
         -o /lcrc/project/SDR/pjiao/data/cnn_train/vx3d/hurricane \
         -d f32 -n 3 -s 100 500 500 --artifact banding --ext f32 --use val 
done