CUDA_VISIBLE_DEVICES=6 python train_cifar_main.py --gpu_id 6 --T 2 --dataset cifar100 --tau 2.0 \
  --model online_spiking_vgg11 \
  --output_dir=$checkpoint --tb --autoaug --cutout --online_update  --drop_rate 0.0 \
  --batch_size 128 --T_max 300 --epochs 400 \
  --optimizer Bop --lr 0.1 --weight_decay 0.0 --threshold 5e-7 --beta1 0.999 --beta2 0.99999 \
  # --resume 1