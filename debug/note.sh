python test_model.py --model dncnn \
    --weight_path /lcrc/project/ECP-EZ/jp/git/arcnn/checkpoints/hurricane_dncnn_no_penalty/best.pth \
    --rel_eb 0.01 \
    --input_size 100 500 500 \
    --patch_size 100 --stride 90 

python test_model.py --model unet \
    --weight_path /lcrc/project/ECP-EZ/jp/git/arcnn/checkpoints/hurricane_unet_no_penalty/best.pth \
    --rel_eb 0.01 \
    --input_size 100 500 500 \
    --patch_size 100 --stride 90 

python test_model.py --model unet \
    --weight_path /lcrc/project/ECP-EZ/jp/git/arcnn/checkpoints/hurricane_unet_no_penalty/best.pth \
    --rel_eb 0.01 \
    --input_size 100 500 500 \
    --patch_size 50 --stride 45