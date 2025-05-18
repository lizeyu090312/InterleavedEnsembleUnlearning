#!/bin/bash

# $1 contains the GPU device
# effects of using different trigger labels.  

# training code
trigger=(0 3 5 8)
attacks=("BadNets-white" "ISSBA" "Smooth")
for attack in ${attacks[@]}; do
    for triggerlabel in ${trigger[@]}; do
        # python train_ensemble_noisy_poison_module.py --poison_logits_noise_variance $variance --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.95 --poison_method "$attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set tiny_imagenet_200 --base_dir "checkpoints/ensemble_no_weight_masking/tiny_imagenet_200"
        python train_ensemble.py --trigger_label $triggerlabel --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.95 --poison_method "$attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_trigger_label/CIFAR10/trigger_label_$triggerlabel"
    done
done

# testing code

parent="./checkpoints/ensemble_trigger_label/CIFAR10"
# List all subdirectories within the parent directory
for triggerlabel in ${trigger[@]}; do
    for attack in ${attacks[@]}; do
        python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/trigger_label_$triggerlabel/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
        python test_ensemble.py --device $1 --arg_file "$parent/trigger_label_$triggerlabel/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    done
done
