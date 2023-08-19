#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --mem=8G
#SBATCH -o log/importance_analysis.out
#SBATCH -e log/importance_analysis.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=48:00:00

eval $(spack load --sh miniconda3)
source activate active-learning

which python3

model_dir=ModelFitting/CNN_Reg/best_model/
dirname=ImportanceAnalysis/MotifCombos/
mkdir -p ${dirname}/{First,Second}Set/
python3 src/importance_analysis.py "${model_dir}" "${dirname}"/FirstSet/
python3 src/importance_analysis.py "${model_dir}" "${dirname}"/SecondSet/ --valid_positions 8 28 48 68 88 108 128

