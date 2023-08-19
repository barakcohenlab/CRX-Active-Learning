#!/bin/bash
# Wrapper script to fit a regression CNN with multiple random starts and then summarize the result
jid=$(sbatch src/cnn_regression_multi_starts.sh)
echo $jid
jid=$(echo $jid | cut -f 4 -d " ")
dirname=ModelFitting/CNN_Reg/
stdout=log/fit_summarize_regression_starts.out
stderr=log/fit_summarize_regression_starts.err
sbatch --dependency=afterok:$jid -o $stdout -e $stderr --mail-type=END,FAIL --wrap="spack load miniconda3; source activate active-learning; python3 src/summarize_regression_starts.py $dirname; python3 src/eval_regression_on_test_sets.py $dirname/best_model/"
