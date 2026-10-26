#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=20G
#SBATCH -p normal
#SBATCH -A hpc-prf-dedsm
#SBATCH -J "Run_tests_%A_%a"
#SBATCH -o "%A_%a_Run_tests.out"

module load tools/Apptainer/1.1.9-GCCcore-11.3.0

echo "Starting job $SLURM_JOB_ID on $(hostname) at $(date)"

apptainer exec --contain --cleanenv --bind "$PWD:$PWD" --pwd "$PWD" python311.sif -m pytest -n auto 
