#!/bin/bash

# Seeing how interleaved unlearning performs on artificial D^{ul} (unlearn sets) with differnt isolation ratios. 
# The poisoning rate is fixed at 0.1

# for ./checkpoints/ensemble_pure_poisoned_unlearn
attacks=("ISSBA" "BadNets-white" "Smooth")

isolation_ratios=(0.01 0.02 0.05 0.09 0.095 0.1 0.2)
# $1 contains device


attacks=("ISSBA" "BadNets-white" "Smooth")
for attack in ${attacks[@]}; do
    for isolation_ratio in ${isolation_ratios[@]}; do
        python train_ensemble_pure_poisoned_unlearn.py --poisoning_rate 0.1 --device $1 --prefinetuning_epochs 0 --prefinetuning_lr -1 --lr 2e-5 --confidence_thresh -1 --poison_method "$attack" --num_workers 6 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set "$2" --base_dir "checkpoints/ensemble_pure_poisoned_unlearn/$2/isolation_ratio_$isolation_ratio" --isolation_ratio $isolation_ratio
        python train_ensemble_pure_poisoned_unlearn.py --poisoning_rate 0.1 --device $1 --prefinetuning_epochs 0 --prefinetuning_lr -1 --lr 2e-5 --confidence_thresh -1 --poison_method "$attack" --num_workers 6 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set tiny_imagenet_200 --base_dir "checkpoints/ensemble_pure_poisoned_unlearn/tiny_imagenet_200/isolation_ratio_$isolation_ratio" --isolation_ratio $isolation_ratio
    done
done

# testing code:

echo $parent
# List all subdirectories within the parent directory
for isol in ${isolation_ratios[@]}; do
    for attack in ${attacks[@]}; do
        parent="./checkpoints/ensemble_pure_poisoned_unlearn/CIFAR10"
        python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/isolation_ratio_$isol/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
        python test_ensemble.py --device $1 --arg_file "$parent/isolation_ratio_$isol/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
        
        parent="./checkpoints/ensemble_pure_poisoned_unlearn/tiny_imagenet_200"
        python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/isolation_ratio_$isol/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
        python test_ensemble.py --device $1 --arg_file "$parent/isolation_ratio_$isol/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    done
done
