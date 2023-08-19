#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -o log/process_join_annotate_counts.out
#SBATCH -e log/process_join_annotate_counts.err
#SBATCH --mail-type=BEGIN,END,FAIL

eval $(spack load --sh miniconda3)
source activate active-learning
python3 src/process_and_join_counts.py
python3 src/annotate_data.py
