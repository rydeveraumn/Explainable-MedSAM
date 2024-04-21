#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --mem=50g
#SBATCH --tmp=30g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
cd /home/csci5980/dever120/Explainable-MedSam
export PATH=/home/csci5980/dever120/.conda/envs/medsam/bin:$PATH
python -m xai_medsam run-inference -i /panfs/jay/groups/7/csci5980/dever120/Explainable-MedSam/datasets/validation-v2 -o /panfs/jay/groups/7/csci5980/dever120/Explainable-MedSam/datasets/validation-medsam-lite-segs-v3/ --lite_medsam_checkpoint_path /panfs/jay/groups/7/csci5980/dever120/Explainable-MedSam/lite_medsam.pth --attention