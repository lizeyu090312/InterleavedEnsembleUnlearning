#!/bin/bash

# Generate attacker-accessible models for the CAT adaptive attack. Run this first for CAT so that the attacker can generate perturbed poisoned data. 

# $1 contains the GPU device

# training code
datasets=("tiny_imagenet_200" "CIFAR10")
attacks=("BadNets-pattern" "BadNets-white" "Blended" "Smooth" "Trojan-SQ")
for attack in ${attacks[@]}; do
    for dataset in ${datasets[@]}; do
        python cat_adaptive_attack_model_gen.py --device $1 --finetuning_epochs 10 --lr 2e-5 --poison_method "$attack" --num_workers 10 --data_set $dataset --base_dir "checkpoints/adaptive_attack_backdoor_img_only/$dataset"
    done
done
