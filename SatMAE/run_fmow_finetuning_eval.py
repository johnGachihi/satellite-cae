python -m torch.distributed.run \
    main_finetune.py \
    --output_dir ./output_ft \
    --log_dir ./output_ft \
    --batch_size 128 --accum_iter 4 \
    --model vit_base_patch16 --epochs 30 --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 \
    --mixup 0.8 --cutmix 1.0 \
    --num_workers 8 \
    --dataset_type sentinel --nb_classes 62 \
    --train_path '/home/ubuntu/satellite-cae/SatMAE/data/fmow_ft_train.csv' \
    --test_path '/home/ubuntu/satellite-cae/SatMAE/data/test_.csv' \
    --finetune ~/checkpoint-98.pth \
    --eval \
    --resume ~/satellite-cae/SatMAE/output_ft/checkpoint-fs-29.pth \
#    --wandb mae_satellite_finetune \
