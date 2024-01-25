#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import selene_sdk
import selene_sdk.samplers
import glob

from mpra_tools import loaders, modeling
from selene_files import enhancer_resnet_regression
    
def performance(model, sampler, seq_size, gpu):
    if gpu:
        device = torch.device("cuda")
        model.to(device)

    batched, truth = sampler.get_data_and_targets(64)
    truth = truth.flatten()
    probs = modeling.cnn_predict_batches(model, batched, use_cuda=gpu, seq_len=seq_size)

    measurements = dict()
    measurements["roc_auc"] = roc_auc_score(truth, probs)
    measurements["pr_auc"] = average_precision_score(truth,probs)
    
    return measurements


def main(folder, gpu):
    # Init default paramters
    L = 230
    folds = range(1,11)
    modes = ["Entropy", "Random"]

    # Find the fold with the minimum number of rounds
    num_rounds = min([len(glob.glob(os.path.join(folder,str(fold),modes[0],"*"))) for fold in folds])
    rounds = ["round"+str(r+1) for r in range(num_rounds)]

    # Find measurements and parameters
    measurements = ["roc_auc", "pr_auc"]

    # Get the performance metrics for Entropy and Random Models
    scores = []
    for fold in folds:
        for mode in modes:
            for r in rounds:
                base_dir = os.path.join(folder, str(fold), mode, r)
                model_path = os.path.join(base_dir, "best_model.pth.tar")
                test_mat = os.path.join(base_dir, "test.mat")
                
                model = enhancer_resnet_regression.EnhancerResnet(L)
                model.output = nn.Sequential(model.output, nn.Sigmoid())    
                model = loaders.load_cnn(
                    model_path, 
                    model=model, 
                    eval_mode=True
                )

                test_sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
                    filepath=test_mat,
                    sequence_key='sequence',
                    targets_key='activity_bin',
                )
                
                scores.append(
                    performance(model,test_sampler,L,gpu).values()
                )

    MI = pd.MultiIndex.from_product([folds,modes,rounds], names=['Fold','Mode','Round'])
    rounds_df = pd.DataFrame(scores, index=MI, columns=measurements)

    print("Fold", fold, "Done", flush=True)

    # Get the performance models for the rounds of Full learning
    scores = []

    for fold in folds:
        base_dir = os.path.join(folder, str(fold), "Full")
        model_path = os.path.join(base_dir, "best_model.pth.tar")
        test_mat = os.path.join(base_dir, "test.mat")
        
        model = enhancer_resnet_regression.EnhancerResnet(L)
        model.output = nn.Sequential(model.output, nn.Sigmoid())    
        model = loaders.load_cnn(
            model_path, 
            model=model, 
            eval_mode=True
            )
        
        test_sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
            filepath=test_mat,
            sequence_key='sequence',
            targets_key='activity_bin',
        )
        
        scores.append(
            performance(model, test_sampler, L, gpu).values()
        )

    MI = pd.MultiIndex.from_product([folds,['Full'],['full']])
    full_df = pd.DataFrame(scores, index=MI, columns=measurements)
    measurements_df = pd.concat([rounds_df, full_df])
    loaders.write_data(measurements_df, os.path.join(folder, "performance_summary.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("folder", help="Data location and output")
    parser.add_argument("--gpu", type=bool, default=False)
    args = parser.parse_args()
    
    main(**vars(args))
