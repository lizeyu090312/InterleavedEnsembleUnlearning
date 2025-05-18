#!/bin/bash

# attacks=("BadNets-white" "BadNets-pattern" "ISSBA" "BATT" "Blended" "Trojan-WM" "Trojan-SQ" "Smooth" "l0-inv" "l2-inv")

attacks=("SIG")
# python train_teacher.py --device $1 --poison_method "Refool" --num_workers 2
# python train_ensemble.py --device $1 --poison_method "Refool" --num_workers 2 --saveckp_freq 2

for attack in ${attacks[@]}
do 
# failed_prefinetuning_lr_1e-3_too_high
    # python test_ensemble.py --device $1 --test_with_poison --arg_file "checkpoints/ensemble_all_defended/tiny_imagenet_200/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    # python test_ensemble.py --device $1 --arg_file "checkpoints/ensemble_all_defended/tiny_imagenet_200/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"

    python test_ensemble.py --device $1 --test_with_poison --arg_file "checkpoints/ensemble_all_defended/gtsrb/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    python test_ensemble.py --device $1 --arg_file "checkpoints/ensemble_all_defended/gtsrb/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"

    python test_ensemble.py --device $1 --test_with_poison --arg_file "checkpoints/ensemble_all_defended/CIFAR10/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    python test_ensemble.py --device $1 --arg_file "checkpoints/ensemble_all_defended/CIFAR10/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"

    python test_ensemble.py --device $1 --test_with_poison --arg_file "checkpoints/ensemble_all_defended/tiny_imagenet_200/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    python test_ensemble.py --device $1 --arg_file "checkpoints/ensemble_all_defended/tiny_imagenet_200/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    
    # python test_ensemble.py --device $1 --test_with_poison --arg_file "checkpoints/ensemble_from_checkpoint/CIFAR10/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    # python test_ensemble.py --device $1 --arg_file "checkpoints/ensemble_from_checkpoint/CIFAR10/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
done
