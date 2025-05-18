#!/bin/bash

## produce defended models (defence: IEU). Vanilla code (i.e., not for ablation)
attacks=("BadNets-white" "BadNets-pattern" "ISSBA" "BATT" "Blended" "Trojan-WM" "Trojan-SQ" "Smooth" "l0-inv" "l2-inv" "SIG")
for attack in ${attacks[@]}; do
  # python train_ensemble.py --gamma 1 --device 0 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.95 --poison_method "$attack" --num_workers 6 --saveckp_freq 2 --prefinetuning_saveckp_freq 2 --data_set CIFAR10 --base_dir "checkpoints/ensemble_LGA_all_defended/CIFAR10"
  python train_ensemble.py --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.95 --poison_method "$attack" --num_workers 10 --saveckp_freq 2 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_all_defended/CIFAR10"
  python train_ensemble.py --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.95 --poison_method "$attack" --num_workers 10 --saveckp_freq 2 --prefinetuning_saveckp_freq 100 --data_set tiny_imagenet_200 --base_dir "checkpoints/ensemble_all_defended/tiny_imagenet_200"
  python train_ensemble.py --device $1 --prefinetuning_epochs 5 --prefinetuning_lr 1e-3 --lr 2e-5 --confidence_thresh 0.998 --poison_method "$attack" --num_workers 10 --saveckp_freq 2 --prefinetuning_saveckp_freq 100 --data_set gtsrb --base_dir "checkpoints/ensemble_all_defended/gtsrb"
done