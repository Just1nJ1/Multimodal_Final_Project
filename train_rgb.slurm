#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint="a4000"
#SBATCH --mem="32g"
#SBATCH --output=logs/output-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hj297@rutgers.edu

source ~/.bashrc
source activate mambaa

cd ~/Justin/Multimodal_Final_Project
python TrainEncoder.py "$@"
