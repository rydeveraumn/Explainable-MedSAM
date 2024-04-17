#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --mem=50g
#SBATCH --tmp=30g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
cd /home/csci5980/dever120/Explainable-MedSam
export PATH=/home/csci5980/dever120/miniconda3/envs/medsam/bin:$PATH
python -m xai_medsam.tasks run-inference