#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH -o log/summarize_k562_iterative.out-%A
#SBATCH -e log/summarize_k562_iterative.err-%A

eval $(spack load --sh miniconda3)
source activate active-learning

python3 src/summarize_k562_iterative.py ModelFitting/K562/ManyRounds/init_10000_inc_3000/
python3 src/summarize_k562_iterative.py ModelFitting/K562/ManyRounds/init_4000_inc_2000/
