#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH -J "text_clust"
#SBATCH -p gpu
#SBATCH -A hpc-prf-dedsm

module load lang/Tkinter/3.9.6-GCCcore-11.2.0
module load lang/Python/3.10.4-GCCcore-11.3.0
module load system/CUDA/12.4.1
source /scratch/hpc-prf-dedsm/sussekl/textClust/.venv/bin/activate

python examples/run_rootClust.py
