#!/bin/bash

# effects of using different poisoning rates
attacks=("BadNets-white" "ISSBA")
poi_rates=(0.02 0.05 0.15 0.2)
for attack in ${attacks[@]}; do
    for poi_rate in ${poi_rates[@]}; do
        python train_ensemble_diff_poi_rate.py --poisoning_rate $poi_rate --device $1 --prefinetuning_epochs 0 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.95 --poison_method "$attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_diff_poi_rate_defend/CIFAR10/poisoning_rate_$poi_rate"        
    done
done

# testing
attacks=("BadNets-white" "ISSBA")
poi_rates=(0.02 0.05 0.15 0.2)
for attack in ${attacks[@]}; do
    for poi_rate in ${poi_rates[@]}; do
        python test_ensemble.py --device $1 --test_with_poison --arg_file "checkpoints/ensemble_diff_poi_rate_defend/CIFAR10/poisoning_rate_$poi_rate/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
        python test_ensemble.py --device $1 --arg_file "checkpoints/ensemble_diff_poi_rate_defend/CIFAR10/poisoning_rate_$poi_rate/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    done
done
