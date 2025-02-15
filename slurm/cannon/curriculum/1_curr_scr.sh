#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-16:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_curriculum_1 \
    training.load_checkpoint_run_id=null \
    furniture=round_table \
    data.dataloader_workers=16 \
    wandb.name=scratch-partial-1
