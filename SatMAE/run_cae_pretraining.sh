python3 -m torch.distributed.run \
    --local_rank 0 \
    main_pretrain.py \
    --model_type cae \
    --batch_size 64 --accum_iter 16 \
    --norm_pix_loss --epochs 100 \
    --blr 1.5e-4 --mask_ratio 0.75 \
    --input_size 224 --patch_size 16 \
    --model mae_vit_base_patch16 \
    --dataset_type sentinel \
    --train_path '/home/ubuntu/satellite-cae/SatMAE/data/sampled_by_location.csv' \
    --wandb 'mae-satellite-pretrain' \
#    --resume output_dir/checkpoint-95.pth
#    --device cpu
#    --output_dir ./data/fmow-sentinel/\
#    --log_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
#    --num_workers 8
