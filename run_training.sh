#!/bin/bash

python src/train_diffusion.py \
  --data-dir @mnist \
  --batch-size 64 \
  --epochs 50 \
  --lr 2e-4 \
  --base-channels 128 \
  --image-size 28 \
  --timesteps 1000 \
  --beta-start 1e-4 \
  --beta-end 0.02 \
  --sample-every 2000 \
  --checkpoint-every 1000 \
  --log-every 50 \
  --eval-every 1 \
  --eval-batches 50 \
  --checkpoint-epoch-every 5 \
  --output-dir trained_model \
  --run-name mnist-ddpm-baseline \
  --seed 3407