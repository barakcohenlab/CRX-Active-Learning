#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --mem=8G
#SBATCH --time=4-12
#SBATCH -o log/cnn_multiclass_cross_validation.out
#SBATCH -e log/cnn_multiclass_cross_validation.err
#SBATCH --mail-type=BEGIN,END,FAIL

eval $(spack load --sh miniconda3)
source activate active-learning

dirname=ModelFitting/CNN_Clf
mkdir -p $dirname/Round3bFull

python3 src/cnn_multiclass_cross_validation.py $dirname
python3 src/cnn_multiclass_cross_validation.py $dirname/Round3bFull --nfolds 1 --modeling_splits Data/round_3b_only.txt --report_train
