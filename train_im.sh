CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29700 \
    train_im.py \
    --gpu_id '0,1,2,3' \
    --T 6 \
    --dataset imagenet \
    --tau 2.0 \
    --model birealnet18 \
    --output_dir=$checkpoint \
    --tb \
    --autoaug \
    --cutout \
    --online_update \
    --drop_rate 0.0 \
    --j 12 \
    --batch_size 256 \
    --T_max 100 \
    --epochs 100 \
    --optimizer Bop \
    --lr 0.2 \
    --weight_decay 0.0 \
    --threshold 1e-6 \
    --beta1 0.999 \
    --beta2 0.99999 \
    # --resume 1