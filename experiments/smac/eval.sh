#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo_eval
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=output/smac/%a.out
#SBATCH --array=0-101

experiment=("best_config_loo_NTSO_seed_0.json --output-csv smac3_results_eval/best_config_loo_NTSO_seed_0.json" \
"best_config_loo_NTSO_seed_1.json --output-csv smac3_results_eval/best_config_loo_NTSO_seed_1.json" \
"best_config_loo_NTSO_seed_2.json --output-csv smac3_results_eval/best_config_loo_NTSO_seed_2.json" \
"best_config_loo_NTSO_seed_3.json --output-csv smac3_results_eval/best_config_loo_NTSO_seed_3.json" \
"best_config_loo_NTSO_seed_4.json --output-csv smac3_results_eval/best_config_loo_NTSO_seed_4.json" \
"best_config_loo_NTSO_seed_5.json --output-csv smac3_results_eval/best_config_loo_NTSO_seed_5.json" \
"best_config_loo_NTSO_seed_6.json --output-csv smac3_results_eval/best_config_loo_NTSO_seed_6.json" \
"best_config_loo_NTSO_seed_7.json --output-csv smac3_results_eval/best_config_loo_NTSO_seed_7.json" \
"best_config_loo_NTSO_seed_8.json --output-csv smac3_results_eval/best_config_loo_NTSO_seed_8.json" \
"best_config_loo_NTSO_seed_9.json --output-csv smac3_results_eval/best_config_loo_NTSO_seed_9.json" \
"best_config_loo_NTS_seed_0.json --output-csv smac3_results_eval/best_config_loo_NTS_seed_0.json" \
"best_config_loo_NTS_seed_1.json --output-csv smac3_results_eval/best_config_loo_NTS_seed_1.json" \
"best_config_loo_NTS_seed_2.json --output-csv smac3_results_eval/best_config_loo_NTS_seed_2.json" \
"best_config_loo_NTS_seed_3.json --output-csv smac3_results_eval/best_config_loo_NTS_seed_3.json" \
"best_config_loo_NTS_seed_4.json --output-csv smac3_results_eval/best_config_loo_NTS_seed_4.json" \
"best_config_loo_NTS_seed_5.json --output-csv smac3_results_eval/best_config_loo_NTS_seed_5.json" \
"best_config_loo_NTS_seed_6.json --output-csv smac3_results_eval/best_config_loo_NTS_seed_6.json" \
"best_config_loo_NTS_seed_7.json --output-csv smac3_results_eval/best_config_loo_NTS_seed_7.json" \
"best_config_loo_NTS_seed_8.json --output-csv smac3_results_eval/best_config_loo_NTS_seed_8.json" \
"best_config_loo_NTS_seed_9.json --output-csv smac3_results_eval/best_config_loo_NTS_seed_9.json" \
"best_config_loo_NT_seed_0.json --output-csv smac3_results_eval/best_config_loo_NT_seed_0.json" \
"best_config_loo_NT_seed_1.json --output-csv smac3_results_eval/best_config_loo_NT_seed_1.json" \
"best_config_loo_NT_seed_2.json --output-csv smac3_results_eval/best_config_loo_NT_seed_2.json" \
"best_config_loo_NT_seed_3.json --output-csv smac3_results_eval/best_config_loo_NT_seed_3.json" \
"best_config_loo_NT_seed_4.json --output-csv smac3_results_eval/best_config_loo_NT_seed_4.json" \
"best_config_loo_NT_seed_5.json --output-csv smac3_results_eval/best_config_loo_NT_seed_5.json" \
"best_config_loo_NT_seed_6.json --output-csv smac3_results_eval/best_config_loo_NT_seed_6.json" \
"best_config_loo_NT_seed_7.json --output-csv smac3_results_eval/best_config_loo_NT_seed_7.json" \
"best_config_loo_NT_seed_8.json --output-csv smac3_results_eval/best_config_loo_NT_seed_8.json" \
"best_config_loo_NT_seed_9.json --output-csv smac3_results_eval/best_config_loo_NT_seed_9.json" \
"best_config_loo_News-T_seed_0.json --output-csv smac3_results_eval/best_config_loo_News-T_seed_0.json" \
"best_config_loo_News-T_seed_1.json --output-csv smac3_results_eval/best_config_loo_News-T_seed_1.json" \
"best_config_loo_News-T_seed_2.json --output-csv smac3_results_eval/best_config_loo_News-T_seed_2.json" \
"best_config_loo_News-T_seed_3.json --output-csv smac3_results_eval/best_config_loo_News-T_seed_3.json" \
"best_config_loo_News-T_seed_4.json --output-csv smac3_results_eval/best_config_loo_News-T_seed_4.json" \
"best_config_loo_News-T_seed_5.json --output-csv smac3_results_eval/best_config_loo_News-T_seed_5.json" \
"best_config_loo_News-T_seed_6.json --output-csv smac3_results_eval/best_config_loo_News-T_seed_6.json" \
"best_config_loo_News-T_seed_7.json --output-csv smac3_results_eval/best_config_loo_News-T_seed_7.json" \
"best_config_loo_News-T_seed_8.json --output-csv smac3_results_eval/best_config_loo_News-T_seed_8.json" \
"best_config_loo_News-T_seed_9.json --output-csv smac3_results_eval/best_config_loo_News-T_seed_9.json" \
"best_config_loo_SO-T_seed_0.json --output-csv smac3_results_eval/best_config_loo_SO-T_seed_0.json" \
"best_config_loo_SO-T_seed_1.json --output-csv smac3_results_eval/best_config_loo_SO-T_seed_1.json" \
"best_config_loo_SO-T_seed_2.json --output-csv smac3_results_eval/best_config_loo_SO-T_seed_2.json" \
"best_config_loo_SO-T_seed_3.json --output-csv smac3_results_eval/best_config_loo_SO-T_seed_3.json" \
"best_config_loo_SO-T_seed_4.json --output-csv smac3_results_eval/best_config_loo_SO-T_seed_4.json" \
"best_config_loo_SO-T_seed_5.json --output-csv smac3_results_eval/best_config_loo_SO-T_seed_5.json" \
"best_config_loo_SO-T_seed_6.json --output-csv smac3_results_eval/best_config_loo_SO-T_seed_6.json" \
"best_config_loo_SO-T_seed_7.json --output-csv smac3_results_eval/best_config_loo_SO-T_seed_7.json" \
"best_config_loo_SO-T_seed_8.json --output-csv smac3_results_eval/best_config_loo_SO-T_seed_8.json" \
"best_config_loo_SO-T_seed_9.json --output-csv smac3_results_eval/best_config_loo_SO-T_seed_9.json" \
"best_config_loo_Semantic10-T_seed_0.json --output-csv smac3_results_eval/best_config_loo_Semantic10-T_seed_0.json" \
"best_config_loo_Semantic10-T_seed_1.json --output-csv smac3_results_eval/best_config_loo_Semantic10-T_seed_1.json" \
"best_config_loo_Semantic10-T_seed_2.json --output-csv smac3_results_eval/best_config_loo_Semantic10-T_seed_2.json" \
"best_config_loo_Semantic10-T_seed_3.json --output-csv smac3_results_eval/best_config_loo_Semantic10-T_seed_3.json" \
"best_config_loo_Semantic10-T_seed_4.json --output-csv smac3_results_eval/best_config_loo_Semantic10-T_seed_4.json" \
"best_config_loo_Semantic10-T_seed_5.json --output-csv smac3_results_eval/best_config_loo_Semantic10-T_seed_5.json" \
"best_config_loo_Semantic10-T_seed_6.json --output-csv smac3_results_eval/best_config_loo_Semantic10-T_seed_6.json" \
"best_config_loo_Semantic10-T_seed_7.json --output-csv smac3_results_eval/best_config_loo_Semantic10-T_seed_7.json" \
"best_config_loo_Semantic10-T_seed_8.json --output-csv smac3_results_eval/best_config_loo_Semantic10-T_seed_8.json" \
"best_config_loo_Semantic10-T_seed_9.json --output-csv smac3_results_eval/best_config_loo_Semantic10-T_seed_9.json" \
"best_config_loo_Semantic40-T_seed_0.json --output-csv smac3_results_eval/best_config_loo_Semantic40-T_seed_0.json" \
"best_config_loo_Semantic40-T_seed_1.json --output-csv smac3_results_eval/best_config_loo_Semantic40-T_seed_1.json" \
"best_config_loo_Semantic40-T_seed_2.json --output-csv smac3_results_eval/best_config_loo_Semantic40-T_seed_2.json" \
"best_config_loo_Semantic40-T_seed_3.json --output-csv smac3_results_eval/best_config_loo_Semantic40-T_seed_3.json" \
"best_config_loo_Semantic40-T_seed_4.json --output-csv smac3_results_eval/best_config_loo_Semantic40-T_seed_4.json" \
"best_config_loo_Semantic40-T_seed_5.json --output-csv smac3_results_eval/best_config_loo_Semantic40-T_seed_5.json" \
"best_config_loo_Semantic40-T_seed_6.json --output-csv smac3_results_eval/best_config_loo_Semantic40-T_seed_6.json" \
"best_config_loo_Semantic40-T_seed_7.json --output-csv smac3_results_eval/best_config_loo_Semantic40-T_seed_7.json" \
"best_config_loo_Semantic40-T_seed_8.json --output-csv smac3_results_eval/best_config_loo_Semantic40-T_seed_8.json" \
"best_config_loo_Semantic40-T_seed_9.json --output-csv smac3_results_eval/best_config_loo_Semantic40-T_seed_9.json" \
"best_config_loo_Semantic5-T_seed_0.json --output-csv smac3_results_eval/best_config_loo_Semantic5-T_seed_0.json" \
"best_config_loo_Semantic5-T_seed_1.json --output-csv smac3_results_eval/best_config_loo_Semantic5-T_seed_1.json" \
"best_config_loo_Semantic5-T_seed_2.json --output-csv smac3_results_eval/best_config_loo_Semantic5-T_seed_2.json" \
"best_config_loo_Semantic5-T_seed_3.json --output-csv smac3_results_eval/best_config_loo_Semantic5-T_seed_3.json" \
"best_config_loo_Semantic5-T_seed_4.json --output-csv smac3_results_eval/best_config_loo_Semantic5-T_seed_4.json" \
"best_config_loo_Semantic5-T_seed_5.json --output-csv smac3_results_eval/best_config_loo_Semantic5-T_seed_5.json" \
"best_config_loo_Semantic5-T_seed_6.json --output-csv smac3_results_eval/best_config_loo_Semantic5-T_seed_6.json" \
"best_config_loo_Semantic5-T_seed_7.json --output-csv smac3_results_eval/best_config_loo_Semantic5-T_seed_7.json" \
"best_config_loo_Semantic5-T_seed_8.json --output-csv smac3_results_eval/best_config_loo_Semantic5-T_seed_8.json" \
"best_config_loo_Semantic5-T_seed_9.json --output-csv smac3_results_eval/best_config_loo_Semantic5-T_seed_9.json" \
"best_config_loo_Trends-T_seed_0.json --output-csv smac3_results_eval/best_config_loo_Trends-T_seed_0.json" \
"best_config_loo_Trends-T_seed_1.json --output-csv smac3_results_eval/best_config_loo_Trends-T_seed_1.json" \
"best_config_loo_Trends-T_seed_2.json --output-csv smac3_results_eval/best_config_loo_Trends-T_seed_2.json" \
"best_config_loo_Trends-T_seed_3.json --output-csv smac3_results_eval/best_config_loo_Trends-T_seed_3.json" \
"best_config_loo_Trends-T_seed_4.json --output-csv smac3_results_eval/best_config_loo_Trends-T_seed_4.json" \
"best_config_loo_Trends-T_seed_5.json --output-csv smac3_results_eval/best_config_loo_Trends-T_seed_5.json" \
"best_config_loo_Trends-T_seed_6.json --output-csv smac3_results_eval/best_config_loo_Trends-T_seed_6.json" \
"best_config_loo_Trends-T_seed_7.json --output-csv smac3_results_eval/best_config_loo_Trends-T_seed_7.json" \
"best_config_loo_Trends-T_seed_8.json --output-csv smac3_results_eval/best_config_loo_Trends-T_seed_8.json" \
"best_config_loo_Trends-T_seed_9.json --output-csv smac3_results_eval/best_config_loo_Trends-T_seed_9.json" \
"best_config_loo_Tweets-T_seed_0.json --output-csv smac3_results_eval/best_config_loo_Tweets-T_seed_0.json" \
"best_config_loo_Tweets-T_seed_1.json --output-csv smac3_results_eval/best_config_loo_Tweets-T_seed_1.json" \
"best_config_loo_Tweets-T_seed_2.json --output-csv smac3_results_eval/best_config_loo_Tweets-T_seed_2.json" \
"best_config_loo_Tweets-T_seed_3.json --output-csv smac3_results_eval/best_config_loo_Tweets-T_seed_3.json" \
"best_config_loo_Tweets-T_seed_4.json --output-csv smac3_results_eval/best_config_loo_Tweets-T_seed_4.json" \
"best_config_loo_Tweets-T_seed_5.json --output-csv smac3_results_eval/best_config_loo_Tweets-T_seed_5.json" \
"best_config_loo_Tweets-T_seed_6.json --output-csv smac3_results_eval/best_config_loo_Tweets-T_seed_6.json" \
"best_config_loo_Tweets-T_seed_7.json --output-csv smac3_results_eval/best_config_loo_Tweets-T_seed_7.json" \
"best_config_loo_Tweets-T_seed_8.json --output-csv smac3_results_eval/best_config_loo_Tweets-T_seed_8.json" \
"best_config_loo_Tweets-T_seed_9.json --output-csv smac3_results_eval/best_config_loo_Tweets-T_seed_9.json" \
"--use-default-config")

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"

module load lang/Python/3.11.3-GCCcore-12.3.0
source /pc2/groups/hpc-prf-dedsm/sussekl/textClustEnv/bin/activate

export SENTENCE_TRANSFORMERS_HOME=/pc2/groups/hpc-prf-dedsm/sussekl/sentence_transformers_cache

COMMANDS_PER_TASK=1
TOTAL_COMMANDS=${#experiment[@]}
task_start=$(( SLURM_ARRAY_TASK_ID * COMMANDS_PER_TASK ))
task_end=$(( task_start + COMMANDS_PER_TASK - 1 ))
for (( cmd_index=task_start; cmd_index<=task_end && cmd_index<TOTAL_COMMANDS; cmd_index++ )); do
    python -u experiments/smac/eval_best_config.py ${experiment[$cmd_index]}
done