#!/bin/bash
#SBATCH --time=7-12
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=output/smac/%a.out
#SBATCH --array=0-109

experiment=( "--loo_dataset Semantic10-T --seed 0" \
"--loo_dataset Semantic10-T --seed 1" \
"--loo_dataset Semantic10-T --seed 2" \
"--loo_dataset Semantic10-T --seed 3" \
"--loo_dataset Semantic10-T --seed 4" \
"--loo_dataset Semantic10-T --seed 5" \
"--loo_dataset Semantic10-T --seed 6" \
"--loo_dataset Semantic10-T --seed 7" \
"--loo_dataset Semantic10-T --seed 8" \
"--loo_dataset Semantic10-T --seed 9" \
"--loo_dataset Semantic40-T --seed 0" \
"--loo_dataset Semantic40-T --seed 1" \
"--loo_dataset Semantic40-T --seed 2" \
"--loo_dataset Semantic40-T --seed 3" \
"--loo_dataset Semantic40-T --seed 4" \
"--loo_dataset Semantic40-T --seed 5" \
"--loo_dataset Semantic40-T --seed 6" \
"--loo_dataset Semantic40-T --seed 7" \
"--loo_dataset Semantic40-T --seed 8" \
"--loo_dataset Semantic40-T --seed 9" \
"--loo_dataset Semantic5-T --seed 0" \
"--loo_dataset Semantic5-T --seed 1" \
"--loo_dataset Semantic5-T --seed 2" \
"--loo_dataset Semantic5-T --seed 3" \
"--loo_dataset Semantic5-T --seed 4" \
"--loo_dataset Semantic5-T --seed 5" \
"--loo_dataset Semantic5-T --seed 6" \
"--loo_dataset Semantic5-T --seed 7" \
"--loo_dataset Semantic5-T --seed 8" \
"--loo_dataset Semantic5-T --seed 9" \
"--loo_dataset NT --seed 0" \
"--loo_dataset NT --seed 1" \
"--loo_dataset NT --seed 2" \
"--loo_dataset NT --seed 3" \
"--loo_dataset NT --seed 4" \
"--loo_dataset NT --seed 5" \
"--loo_dataset NT --seed 6" \
"--loo_dataset NT --seed 7" \
"--loo_dataset NT --seed 8" \
"--loo_dataset NT --seed 9" \
"--loo_dataset NTS --seed 0" \
"--loo_dataset NTS --seed 1" \
"--loo_dataset NTS --seed 2" \
"--loo_dataset NTS --seed 3" \
"--loo_dataset NTS --seed 4" \
"--loo_dataset NTS --seed 5" \
"--loo_dataset NTS --seed 6" \
"--loo_dataset NTS --seed 7" \
"--loo_dataset NTS --seed 8" \
"--loo_dataset NTS --seed 9" \
"--loo_dataset NTSO --seed 0" \
"--loo_dataset NTSO --seed 1" \
"--loo_dataset NTSO --seed 2" \
"--loo_dataset NTSO --seed 3" \
"--loo_dataset NTSO --seed 4" \
"--loo_dataset NTSO --seed 5" \
"--loo_dataset NTSO --seed 6" \
"--loo_dataset NTSO --seed 7" \
"--loo_dataset NTSO --seed 8" \
"--loo_dataset NTSO --seed 9" \
"--loo_dataset SO-T --seed 0" \
"--loo_dataset SO-T --seed 1" \
"--loo_dataset SO-T --seed 2" \
"--loo_dataset SO-T --seed 3" \
"--loo_dataset SO-T --seed 4" \
"--loo_dataset SO-T --seed 5" \
"--loo_dataset SO-T --seed 6" \
"--loo_dataset SO-T --seed 7" \
"--loo_dataset SO-T --seed 8" \
"--loo_dataset SO-T --seed 9" \
"--loo_dataset News-T --seed 0" \
"--loo_dataset News-T --seed 1" \
"--loo_dataset News-T --seed 2" \
"--loo_dataset News-T --seed 3" \
"--loo_dataset News-T --seed 4" \
"--loo_dataset News-T --seed 5" \
"--loo_dataset News-T --seed 6" \
"--loo_dataset News-T --seed 7" \
"--loo_dataset News-T --seed 8" \
"--loo_dataset News-T --seed 9" \
"--loo_dataset Trends-T --seed 0" \
"--loo_dataset Trends-T --seed 1" \
"--loo_dataset Trends-T --seed 2" \
"--loo_dataset Trends-T --seed 3" \
"--loo_dataset Trends-T --seed 4" \
"--loo_dataset Trends-T --seed 5" \
"--loo_dataset Trends-T --seed 6" \
"--loo_dataset Trends-T --seed 7" \
"--loo_dataset Trends-T --seed 8" \
"--loo_dataset Trends-T --seed 9" \
"--loo_dataset Tweets-T --seed 0" \
"--loo_dataset Tweets-T --seed 1" \
"--loo_dataset Tweets-T --seed 2" \
"--loo_dataset Tweets-T --seed 3" \
"--loo_dataset Tweets-T --seed 4" \
"--loo_dataset Tweets-T --seed 5" \
"--loo_dataset Tweets-T --seed 6" \
"--loo_dataset Tweets-T --seed 7" \
"--loo_dataset Tweets-T --seed 8" \
"--loo_dataset Tweets-T --seed 9" )

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"

module load lang/Python/3.11.3-GCCcore-12.3.0
source /scratch/hpc-prf-dedsm/sussekl/textClust/.textClustEnv/bin/activate

COMMANDS_PER_TASK=1
TOTAL_COMMANDS=${#experiment[@]}
task_start=$(( SLURM_ARRAY_TASK_ID * COMMANDS_PER_TASK ))
task_end=$(( task_start + COMMANDS_PER_TASK - 1 ))
for (( cmd_index=task_start; cmd_index<=task_end && cmd_index<TOTAL_COMMANDS; cmd_index++ )); do
    python experiments/smac/optimize.py ${experiment[$cmd_index]}
done