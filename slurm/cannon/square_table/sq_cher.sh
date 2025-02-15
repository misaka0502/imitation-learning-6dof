#!/bin/bash

#SBATCH -p gpu                
#SBATCH -t 2-00:00  
#SBATCH --mem=256G            
#SBATCH --gres=gpu:1          
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log   

python -m src.train.bc_no_rollout \
	+experiment=image_cherry_noise \
	furniture=square_table \
	data.dataloader_workers=16
