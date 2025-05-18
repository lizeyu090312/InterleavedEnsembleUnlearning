#!/bin/bash

# $1 contains the GPU device, $2 the vit arch for student, $3 for whether student or teacher.
# "student" is training with the IEU defence. "teacher" means training without any defence

# finetuning ensemble:
attacks=("BadNets-white" "ISSBA" "Smooth" "BATT")
vitarches=("cait" "deit" "xcit" "pit" "vit" "ResNet-18" "WRN-50-2")
if [ "$3" == "student" ]; then
    for attack in ${attacks[@]}; do
        python train_ensemble_diff_vit_arch.py --model_arch $2 --device $1 --prefinetuning_epochs 0 --finetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.95 --poison_method "$attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_diff_vit_arch/CIFAR10/vit_arch_$2" --seed 24221
    done
elif [ "$3" == "teacher" ]; then
# finetuning teacher ($2 is not used):
    vitarches=("vgg11")
    for vitarch in ${vitarches[@]}; do
        python train_teacher_diff_vit_arch.py --model_arch $vitarch --device $1 --lr 2e-5 --poison_method "BadNets-white" --num_workers 10 --data_set CIFAR10 --base_dir "checkpoints/ensemble_diff_vit_arch/CIFAR10/vit_arch_$vitarch"
    done
elif [ "$3" == "no-attack" ]; then
# no-attack, $2 not used
    vitarches=("cait" "deit" "xcit" "pit" "vit" "ResNet-18")
    for vitarch in ${vitarches[@]}; do
        python train_ensemble_diff_vit_arch.py --model_arch $vitarch --device $1 --prefinetuning_epochs 10 --finetuning_epochs 10 --prefinetuning_lr 2e-4 --lr 2e-5 --confidence_thresh 0.99 --poison_method "no-attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set CIFAR10 --base_dir "checkpoints/ensemble_diff_vit_arch/CIFAR10/vit_arch_$vitarch" --finetune_with_no_poison
    done
fi


# testing student with attack:
attacks=("BadNets-white" "ISSBA" "Smooth" "BATT")
parent="./checkpoints/ensemble_diff_vit_arch/CIFAR10"
echo $parent
for vitarch in ${vitarches[@]}; do
  for attack in ${attacks[@]}; do
    python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/vit_arch_$vitarch/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    python test_ensemble.py --device $1 --arg_file "$parent/vit_arch_$vitarch/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
  done
done

# testing student with no-attack
for vitarch in ${vitarches[@]}; do
    python test_ensemble.py --device $1 --arg_file "$parent/vit_arch_$vitarch/ensemble_no-attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
done

# # testing teacher
parent="./checkpoints/ensemble_diff_vit_arch/CIFAR10"
for vitarch in ${vitarches[@]}; do
  python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/vit_arch_$vitarch/teacher_BadNets-white/commandline_args.txt" --write_to_log --student_or_teacher "teacher"
  python test_ensemble.py --device $1 --arg_file "$parent/vit_arch_$vitarch/teacher_BadNets-white/commandline_args.txt" --write_to_log --student_or_teacher "teacher"
done