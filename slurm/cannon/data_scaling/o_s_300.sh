#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-16:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

# Launch the run
python -m src.train.bc_no_rollout \
    +experiment=image_collect_infer \
    furniture=one_leg \
    data.dataloader_workers=16 \
    data.data_subset=250 \
    wandb.project=one_leg-data-scaling-1