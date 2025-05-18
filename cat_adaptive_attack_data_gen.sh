#!/bin/bash


# Generate perturbed poisoned images for the CAT adaptive attack. Run this second for CAT. 

# $1 contains the GPU device

# datagen code
datasets=("CIFAR10" "tiny_imagenet_200")
attacks=("BadNets-pattern" "BadNets-white" "Blended" "Smooth" "Trojan-SQ")

for attack in ${attacks[@]}; do
    for dataset in ${datasets[@]}; do
        python cat_adaptive_attack_data_gen.py --device $1 --poison_method "$attack" --num_workers 10 --data_set $dataset --base_dir "checkpoints/adaptive_attack_backdoor_img_only/$dataset"
    done
done
