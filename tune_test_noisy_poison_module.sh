#!/bin/bash

# $1 contains the GPU device

# effect of adding noise to y_hat_p during stage 2

# training code
variances=(0.01 0.1 0.5 1.0 2.0)
attacks=("BadNets-white" "ISSBA" "Smooth" "BATT")
for attack in ${attacks[@]}; do
    for variance in ${variances[@]}; do
        python train_ensemble_noisy_poison_module.py --poison_logits_noise_variance $variance --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.95 --poison_method "$attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_noisy_poison_module/CIFAR10/variance_$variance"
    done
done

# testing code:
attacks=("BATT" "BadNets-white" "ISSBA" "Smooth")
variances=(0.01 0.1 0.5 1.0 2.0)
parent="./checkpoints/ensemble_noisy_poison_module/CIFAR10"
for variance in ${variances[@]}; do
  for attack in ${attacks[@]}; do
    python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/variance_$variance/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    python test_ensemble.py --device $1 --arg_file "$parent/variance_$variance/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
  done
done