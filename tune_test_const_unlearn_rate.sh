#!/bin/bash

# testing the effects of using a constant unlearning rate
attacks=("BATT" "BadNets-white" "ISSBA" "l0-inv" "Smooth" "Trojan-WM")
uls=(1 2 4)

# $1 stores device
for ul in ${uls[@]}; do
  for attack in ${attacks[@]}; do
    python train_ensemble_const_unlearn_rate.py --scale_factor $ul --confidence_thresh 0.95 --device $1 --prefinetuning_epochs 0 --prefinetuning_lr 2e-4 --lr 2e-5 --poison_method "$attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_const_unlearn_rate/CIFAR10/scale_factor_${ul}"
    python train_ensemble_const_unlearn_rate.py --scale_factor $ul --confidence_thresh 0.95 --device $1 --prefinetuning_epochs 0 --prefinetuning_lr 2e-4 --lr 2e-5 --poison_method "$attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set tiny_imagenet_200 --base_dir "checkpoints/ensemble_const_unlearn_rate/tiny_imagenet_200/scale_factor_${ul}"
  done
done

# testing code not needed since file train_ensemble_const_unlearn_rate has testing