#!/bin/bash

# $1 contains the GPU device
# testing if equation (1) in the paper is useful by setting y_hat = y_hat_r instead of what's specified in equation (1)
# training code
attacks=("BadNets-white" "ISSBA" "Smooth" "BATT")
for attack in ${attacks[@]}; do
  python train_ensemble_no_weight_masking.py --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.95 --poison_method "$attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set tiny_imagenet_200 --base_dir "checkpoints/ensemble_no_weight_masking/tiny_imagenet_200"
  python train_ensemble_no_weight_masking.py --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.95 --poison_method "$attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_no_weight_masking/CIFAR10"
done


# testing code
attacks=("BadNets-white" "ISSBA" "Smooth" "BATT")
for attack in ${attacks[@]}
do 
    python test_ensemble.py --device $1 --test_with_poison --arg_file "checkpoints/ensemble_no_weight_masking/tiny_imagenet_200/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    python test_ensemble.py --device $1 --arg_file "checkpoints/ensemble_no_weight_masking/tiny_imagenet_200/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    
    python test_ensemble.py --device $1 --test_with_poison --arg_file "checkpoints/ensemble_no_weight_masking/CIFAR10/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    python test_ensemble.py --device $1 --arg_file "checkpoints/ensemble_no_weight_masking/CIFAR10/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
done
