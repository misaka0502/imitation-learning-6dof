#!/bin/bash

#SBATCH -p vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH -t 1-08:00
#SBATCH --mem=64G            
#SBATCH --gres=gpu:1          
#SBATCH -c 32
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log

source /data/scratch/ankile/.config/.slurm.env

# Run (default - with chunking)
python -m src.train.bc_no_rollout +experiment=image_mlp_10m furniture=one_leg data.dataloader_workers=32 
# python -m src.train.bc_no_rollout +experiment=image_mlp_10m wandb.mode=offline furniture=round_table data.dataloader_workers=32 
