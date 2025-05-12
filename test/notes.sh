python train_arcnn_residual.py  \
 --input_dir /lcrc/project/SDR/pjiao/data/cnn_train/patches_64_32_orig/input/ \
 --target_dir /lcrc/project/SDR/pjiao/data/cnn_train/patches_64_32_orig/target/ \
 --shape 1 64 64 --dtype f32 --entropy none --train yes  \
 --outputs_dir /lcrc/project/SDR/pjiao/data/cnn_train/patches_64_32_orig/models/ \
 --model dncnn --batch_size 128