# #!/bin/bash

# effects of using LGA/Flooding (see Anti-Backdoor Learning) when used to tune the poisoned module f_p

# for checkpoints/ensemble_LGA_Flood_no_method

attacks=("BadNets-white" "ISSBA" "Smooth")
methods=("LGA" "Flooding")
# $1 is device, $2 is dataset name (e.g., CIFAR10 / tiny_imagenet_200)

gamma_or_flooding=-1
prefinetuning_epochs=-1
if [[ "$2" == "CIFAR10" ]]; then
    gamma_or_flooding=1.5
    prefinetuning_epochs=10
    confidence_thresh=0.95
    prefinetuning_lr=0.0002
elif [[ "$2" == "tiny_imagenet_200" ]]; then
    gamma_or_flooding=3
    prefinetuning_epochs=10
    confidence_thresh=0.95
    prefinetuning_lr=0.0002
elif [[ "$2" == "gtsrb" ]]; then
    gamma_or_flooding=1
    confidence_thresh=0.9
    prefinetuning_epochs=10
    prefinetuning_lr=0.0002
else
    echo "Incorrect dataset name, got dataset == $1, choose from 'tiny_imagenet_200', 'CIFAR10', 'gtsrb'"
    exit
fi

for method in ${methods[@]}; do
    for attack in ${attacks[@]}; do
        python train_ensemble_LGA_or_Flooding.py --base_dir "checkpoints/ensemble_LGA_Flood_no_method/$2/conf_thresh_$confidence_thresh/method_$method" --gradient_ascent_type $method --gamma_or_flooding $gamma_or_flooding --confidence_thresh $confidence_thresh --poisoning_rate 0.1 --device $1 --prefinetuning_epochs $prefinetuning_epochs --finetuning_epochs 10 --prefinetuning_lr $prefinetuning_lr --lr 2e-5 --poison_method "$attack" --num_workers 10 --saveckp_freq 100 --prefinetuning_saveckp_freq 100 --data_set "$2"
    done
done


# results_log generated in train_ensemble_LGA_or_Flooding
# ASR,CA generated using test_ensemble.py
attacks=("BadNets-white" "ISSBA" "Smooth")
methods=("LGA" "Flooding")
parent="./checkpoints/ensemble_LGA_Flood_no_method"
for attack in ${attacks[@]}; do
    for method in ${methods[@]}; do
        python test_ensemble.py --device $1 --test_with_poison --arg_file "$parent/$2/conf_thresh_0.9/method_$method/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
        python test_ensemble.py --device $1 --arg_file "$parent/$2/conf_thresh_0.9/method_$method/ensemble_$attack/commandline_args.txt" --write_to_log --student_or_teacher "student"
    done
done