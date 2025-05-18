#!/bin/bash

# which method is better: isolating a fixed ratio of samples (isolation ratio, found in Anti-Backdoor Learning) 
# or my method (confidence threshold)
poisoning_rates=(0.02 0.05 0.1 0.15 0.2)
for poisoning_rate in ${poisoning_rates[@]}; do
    python train_ensemble_isol_ratio_vs_conf_thresh.py --confidence_thresh 0.95 --isolation_ratio 0.1 --poisoning_rate $poisoning_rate --device 0 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr -1 --poison_method "BadNets-white" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 1 --data_set CIFAR10 --base_dir "checkpoints/ensemble_diff_poisoning_rate/CIFAR10/poisoning_rate_$poisoning_rate"
done


# testing code included in train_ensemble_isol_ratio_vs_conf_thresh