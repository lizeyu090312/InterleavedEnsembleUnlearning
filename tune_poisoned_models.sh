#!/bin/bash

## produces poisoned models
attacks=("BadNets-white" "BadNets-pattern" "ISSBA" "BATT" "Blended" "Trojan-WM" "Trojan-SQ" "Smooth" "l0-inv" "l2-inv" "SIG")

for attack in ${attacks[@]}; do
  python train_teacher.py --device $1 --poison_method "$attack" --num_workers 6 --data_set tiny_imagenet_200 --base_dir "checkpoints/ensemble_from_checkpoint_imagenet"
  python train_teacher.py --device $1 --poison_method "$attack" --num_workers 6 --data_set CIFAR10 --base_dir "checkpoints/ensemble_from_checkpoint_CIFAR10"
  python train_teacher.py --device $1 --poison_method "$attack" --num_workers 6 --data_set gtsrb --base_dir "checkpoints/ensemble_from_checkpoint_gtsrb"
done
