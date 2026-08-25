#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
# Two datasets are processed by each array task.
#SBATCH --array=0-9
#SBATCH -J "textClust"
#SBATCH -p gpu
#SBATCH -A hpc-prf-dedsm

module load lang/Python/3.11.5-GCCcore-13.2.0
module load lib/libffi/3.4.4-GCCcore-13.2.0
module load lang/Tkinter/3.11.5-GCCcore-13.2.0
module load devel/protobuf/25.3-GCCcore-13.2.0
module load system/CUDA/12.8.0
source .textClust_venv/bin/activate

DATASETS=(
	Arabic_7
	Arabic_Twitter
	CrisisLexT6
	CrisisLexT7
	CrisisLexT26
	CrisisMMD
	CrisisNLP
	Event2012
	Event2012_100
	Event2018
	Event2018_100
	HumAID
	KBP
	MAVEN
	Mix_Data
	Semantic5-T
	Semantic5-T_sorted
	Semantic10-T
	Semantic40-T
)

START=$((SLURM_ARRAY_TASK_ID * 2))
python -u src/textClust/preprocessing/offline_preprocessing.py \
	--datasets "${DATASETS[@]:$START:2}"
