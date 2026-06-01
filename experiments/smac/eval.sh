#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo_eval
#SBATCH --cpus-per-task=128
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"

module load lang/Python/3.11.3-GCCcore-12.3.0
source /scratch/hpc-prf-dedsm/sussekl/textClust/.textClustEnv/bin/activate

python experiments/smac/eval_best_config.py