#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=650G
#SBATCH -o log/svm_multiclass_cross_validation.out
#SBATCH -e log/svm_multiclass_cross_validation.err
#SBATCH --mail-type=BEGIN,END,FAIL

eval $(spack load --sh miniconda3)
source activate active-learning

if [[ $# -eq 1 ]] ; then
    dirname=$1
    args="--checkpoint "$dirname
elif [[ $# -eq 0 ]] ; then
    dirname=ModelFitting/SVM/
    mkdir -p "${dirname}"
    args=$dirname
else
    echo "Usage: $(basename $0) [checkpoint_dir]"
    exit 1
fi

python3 src/svm_multiclass_cross_validation.py $args
