#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=output/smac/%a.out
#SBATCH --array=0-109

experiment=( "--dataset Semantic10-T --smac_seed 0" \
"--dataset Semantic10-T --smac_seed 1" \
"--dataset Semantic10-T --smac_seed 2" \
"--dataset Semantic10-T --smac_seed 3" \
"--dataset Semantic10-T --smac_seed 4" \
"--dataset Semantic10-T --smac_seed 5" \
"--dataset Semantic10-T --smac_seed 6" \
"--dataset Semantic10-T --smac_seed 7" \
"--dataset Semantic10-T --smac_seed 8" \
"--dataset Semantic10-T --smac_seed 9" \
"--dataset Semantic40-T --smac_seed 0" \
"--dataset Semantic40-T --smac_seed 1" \
"--dataset Semantic40-T --smac_seed 2" \
"--dataset Semantic40-T --smac_seed 3" \
"--dataset Semantic40-T --smac_seed 4" \
"--dataset Semantic40-T --smac_seed 5" \
"--dataset Semantic40-T --smac_seed 6" \
"--dataset Semantic40-T --smac_seed 7" \
"--dataset Semantic40-T --smac_seed 8" \
"--dataset Semantic40-T --smac_seed 9" \
"--dataset Semantic5-T --smac_seed 0" \
"--dataset Semantic5-T --smac_seed 1" \
"--dataset Semantic5-T --smac_seed 2" \
"--dataset Semantic5-T --smac_seed 3" \
"--dataset Semantic5-T --smac_seed 4" \
"--dataset Semantic5-T --smac_seed 5" \
"--dataset Semantic5-T --smac_seed 6" \
"--dataset Semantic5-T --smac_seed 7" \
"--dataset Semantic5-T --smac_seed 8" \
"--dataset Semantic5-T --smac_seed 9" \
"--dataset NT --smac_seed 0" \
"--dataset NT --smac_seed 1" \
"--dataset NT --smac_seed 2" \
"--dataset NT --smac_seed 3" \
"--dataset NT --smac_seed 4" \
"--dataset NT --smac_seed 5" \
"--dataset NT --smac_seed 6" \
"--dataset NT --smac_seed 7" \
"--dataset NT --smac_seed 8" \
"--dataset NT --smac_seed 9" \
"--dataset NTS --smac_seed 0" \
"--dataset NTS --smac_seed 1" \
"--dataset NTS --smac_seed 2" \
"--dataset NTS --smac_seed 3" \
"--dataset NTS --smac_seed 4" \
"--dataset NTS --smac_seed 5" \
"--dataset NTS --smac_seed 6" \
"--dataset NTS --smac_seed 7" \
"--dataset NTS --smac_seed 8" \
"--dataset NTS --smac_seed 9" \
"--dataset NTSO --smac_seed 0" \
"--dataset NTSO --smac_seed 1" \
"--dataset NTSO --smac_seed 2" \
"--dataset NTSO --smac_seed 3" \
"--dataset NTSO --smac_seed 4" \
"--dataset NTSO --smac_seed 5" \
"--dataset NTSO --smac_seed 6" \
"--dataset NTSO --smac_seed 7" \
"--dataset NTSO --smac_seed 8" \
"--dataset NTSO --smac_seed 9" \
"--dataset SO-T --smac_seed 0" \
"--dataset SO-T --smac_seed 1" \
"--dataset SO-T --smac_seed 2" \
"--dataset SO-T --smac_seed 3" \
"--dataset SO-T --smac_seed 4" \
"--dataset SO-T --smac_seed 5" \
"--dataset SO-T --smac_seed 6" \
"--dataset SO-T --smac_seed 7" \
"--dataset SO-T --smac_seed 8" \
"--dataset SO-T --smac_seed 9" \
"--dataset News-T --smac_seed 0" \
"--dataset News-T --smac_seed 1" \
"--dataset News-T --smac_seed 2" \
"--dataset News-T --smac_seed 3" \
"--dataset News-T --smac_seed 4" \
"--dataset News-T --smac_seed 5" \
"--dataset News-T --smac_seed 6" \
"--dataset News-T --smac_seed 7" \
"--dataset News-T --smac_seed 8" \
"--dataset News-T --smac_seed 9" \
"--dataset Trends-T --smac_seed 0" \
"--dataset Trends-T --smac_seed 1" \
"--dataset Trends-T --smac_seed 2" \
"--dataset Trends-T --smac_seed 3" \
"--dataset Trends-T --smac_seed 4" \
"--dataset Trends-T --smac_seed 5" \
"--dataset Trends-T --smac_seed 6" \
"--dataset Trends-T --smac_seed 7" \
"--dataset Trends-T --smac_seed 8" \
"--dataset Trends-T --smac_seed 9" \
"--dataset Tweets-T --smac_seed 0" \
"--dataset Tweets-T --smac_seed 1" \
"--dataset Tweets-T --smac_seed 2" \
"--dataset Tweets-T --smac_seed 3" \
"--dataset Tweets-T --smac_seed 4" \
"--dataset Tweets-T --smac_seed 5" \
"--dataset Tweets-T --smac_seed 6" \
"--dataset Tweets-T --smac_seed 7" \
"--dataset Tweets-T --smac_seed 8" \
"--dataset Tweets-T --smac_seed 9" )

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