python -m torch.distributed.run \
    main_finetune.py \
    --output_dir ./output_ft \
    --log_dir ./output_ft \
    --batch_size 4 --accum_iter 4 \
    --model vit_base_patch16 --epochs 50 --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 \
    --mixup 0.8 --cutmix 1.0 \
    --finetune ./output_dir/checkpoint-95.pth \
    --num_workers 8 \
    --dataset_type sentinel \
    --train_path '/home/ubuntu/SatMAE/data/sampled_by_location.csv' \
    --test_path '/home/ubuntu/SatMAE/data/val_.csv' \
    --wandb mae_satellite_finetune
