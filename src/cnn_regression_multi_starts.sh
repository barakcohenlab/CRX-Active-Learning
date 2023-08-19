#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --mem=8G
#SBATCH --array=1-20%3
#SBATCH -o log/cnn_regression_multi_starts-%a.out
#SBATCH -e log/cnn_regression_multi_starts-%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

eval $(spack load --sh miniconda3)
source activate active-learning

dirname=ModelFitting/CNN_Reg/"${SLURM_ARRAY_TASK_ID}"
mkdir -p $dirname
python3 src/cnn_regression_random_start.py "${SLURM_ARRAY_TASK_ID}" "${dirname}"
