#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --mem=50g
#SBATCH --tmp=30g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
cd /home/jusun/dever120/HMS-Harmful-Brain-Activity-Classification
export PATH=/home/jusun/dever120/miniconda3/envs/kaggle/bin:$PATH
python -m tasks run-model-training-progressive-resizing-task-v1