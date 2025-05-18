#!/bin/bash

# effects of changing the confidence threshold (IEU to defend)

attacks=("BATT" "BadNets-white" "ISSBA" "Smooth")
confthresholds=(0.9 0.99)

# $1 stores device
for confthresh in ${confthresholds[@]}; do
  for attack in ${attacks[@]}; do
    python train_ensemble.py --confidence_thresh $confthresh --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --poison_method "$attack" --num_workers 10 --saveckp_freq 2 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_variable_conf_thresh/CIFAR10/conf_thresh_$confthresh"
    python train_ensemble.py --confidence_thresh $confthresh --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --poison_method "$attack" --num_workers 10 --saveckp_freq 2 --prefinetuning_saveckp_freq 100 --data_set tiny_imagenet_200 --base_dir "checkpoints/ensemble_variable_conf_thresh/tiny_imagenet_200/conf_thresh_$confthresh"
  done
done

# testing code:
for confthresh in ${confthresholds[@]}; do
  for attack in ${attacks[@]}; do
    parent="./checkpoints/ensemble_variable_conf_thresh/CIFAR10"
    python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/conf_thresh_$confthresh/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    python test_ensemble.py --device $1 --arg_file "$parent/conf_thresh_$confthresh/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    
    parent="./checkpoints/ensemble_variable_conf_thresh/tiny_imagenet_200"
    python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/conf_thresh_$confthresh/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    python test_ensemble.py --device $1 --arg_file "$parent/conf_thresh_$confthresh/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
  done
done
