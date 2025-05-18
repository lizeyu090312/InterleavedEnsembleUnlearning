#!/bin/bash

# train models using the perturbed poisoned dataset, CAT attack, no defence. Run this after generating the dataset. 
# $1 contains the GPU device

# training code
datasets=("CIFAR10" "tiny_imagenet_200")
attacks=("BadNets-pattern" "BadNets-white" "Blended" "Smooth" "Trojan-SQ")
for attack in ${attacks[@]}; do
    for dataset in ${datasets[@]}; do
        python cat_adaptive_attack_train_teacher.py --device $1 --finetuning_epochs 10 --lr 2e-5 --poison_method "$attack" --num_workers 10 --data_set $dataset --base_dir "checkpoints/adaptive_attack_backdoor_img_only/$dataset"
    done
done


# testing code
for dataset in ${datasets[@]}; do
    for attack in ${attacks[@]}; do
        parent="./checkpoints/adaptive_attack_backdoor_img_only/$dataset"
        echo $parent
        python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/teacher_$attack/commandline_args.txt" --write_to_log --student_or_teacher "teacher"
        python test_ensemble.py --device $1 --arg_file "$parent/teacher_$attack/commandline_args.txt" --write_to_log --student_or_teacher "teacher"
    done
done
