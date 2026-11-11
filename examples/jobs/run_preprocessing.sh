#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-12
#SBATCH -J "textClust"
#SBATCH -p gpu
#SBATCH -A hpc-prf-dedsm

module load tools/Apptainer/1.1.9-GCCcore-11.3.0

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
	NT
	NTS
	NTSO-T
	SO-T
	News-T
	Trends-T
	Tweets-T
)

START=$((SLURM_ARRAY_TASK_ID * 2))
apptainer exec \
	--nv \
	--bind "$PWD:/opt/project" \
	--pwd /opt/project \
	"$PWD/textclust-preprocessing.sif" \
	python -u /opt/project/src/textClust/preprocessing/offline_preprocessing.py \
	--datasets "${DATASETS[@]:$START:2}"
