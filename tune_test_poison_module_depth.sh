#!/bin/bash

# Effects of using a more expressive f_p (poisoned module)

# training code
attacks=("BadNets-white" "ISSBA" "Blended")
# CIFAR10 best hparams
# for poison_module_depth=2, conf_thresh=0.99; for poison_module_depth=3, conf_thresh=0.998

# $2 determines whether to use best hparam or not; $1 is the device.
if [[ "$2" == "use_best_hparam" ]]; then
  conf_thresh_2=0.99
  conf_thresh_3=0.998
else
  conf_thresh_2=0.95
  conf_thresh_3=0.95
fi
echo $conf_thresh_2
echo $conf_thresh_3
for attack in ${attacks[@]}; do
  python train_ensemble_poison_module_depth.py --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --poison_method "$attack" --num_workers 10 --saveckp_freq 2 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_poison_module_depth_best_conf_thresh/CIFAR10/poison_module_depth_2" --poison_module_depth 2 --confidence_thresh $conf_thresh_2 
  python train_ensemble_poison_module_depth.py --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --poison_method "$attack" --num_workers 10 --saveckp_freq 2 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_poison_module_depth_best_conf_thresh/CIFAR10/poison_module_depth_3" --poison_module_depth 3 --confidence_thresh $conf_thresh_3
done

# run the following (tinyimagenet) with fixed confidence threshold. i.e. "$2" ~= "use_best_hparam"
for attack in ${attacks[@]}; do
  python train_ensemble_poison_module_depth.py --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --poison_method "$attack" --num_workers 10 --saveckp_freq 2 --prefinetuning_saveckp_freq 100 --data_set tiny_imagenet_200 --base_dir "checkpoints/ensemble_poison_module_depth/tiny_imagenet_200/poison_module_depth_2" --poison_module_depth 2 --confidence_thresh $conf_thresh_2 
  python train_ensemble_poison_module_depth.py --device $1 --prefinetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --poison_method "$attack" --num_workers 10 --saveckp_freq 2 --prefinetuning_saveckp_freq 100 --data_set tiny_imagenet_200 --base_dir "checkpoints/ensemble_poison_module_depth/tiny_imagenet_200/poison_module_depth_3" --poison_module_depth 3 --confidence_thresh $conf_thresh_3
done


# testing code

poisondepths=(2 3)
# dataset_name is in $2, device is in $1. 
parent="./checkpoints/ensemble_poison_module_depth_best_conf_thresh/$2"
for poisondepth in ${poisondepths[@]}; do
  for attack in ${attacks[@]}; do
    python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/poison_module_depth_$poisondepth/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    python test_ensemble.py --device $1 --arg_file "$parent/poison_module_depth_$poisondepth/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
  done
done
