# The official implementation of *Interleaved Ensemble Unlearning: a Dynamic Training-Time Backdoor Defence*
Under review.

### Introduction
This repository contains the official implementation of *Interleaved Ensemble Unlearning: a Dynamic Training-Time Backdoor Defence*. Please create an environment using conda and install some packages using `pip install -r requirements.txt`. Datasets should be downloaded in `./datasets`. Note: ensure that your environment does not contain the `datasets` module from HuggingFace. 

### Defending using IEU
Please run `bash tune_IEU_defended_models.sh <device>` to see IEU in action on a Vision Transformer. Please download the relevant checkpoints to `./checkpoints` (please refer to `./checkpoints/README.md`). The indices of poisoned images can be chosen in `shuffle_indices.ipynb`. Use `bash tune_poisoned_models.sh <device>` to tune models without any defence. 

Please refer to `tune_test_diff_arch.sh` to see how IEU performs on different architectures (ViT variants and CNNs). 

### Ablation
Please refer to `tune_test_**.sh` for ablation experiments. 

### Defending against an adaptive attack
We use a modifed version of the attack in *Towards reliable backdoor attacks on vision transformers* (https://openreview.net/forum?id=MLShfiJ3CB) as the adaptive attack used in this paper. Please refer to `cat_adaptive_attack**.sh` to run the adaptive attack. First, generate attacker-accessible models using `cat_adaptive_attack_model_gen.sh`. Then, generate the perturbed poisoined data using `cat_adaptive_attack_data_gen.sh`. Finally, tune undefended models and models defended using IEU using `cat_adaptive_attack_train_test_**.sh`. 

### Notes about the TinyImageNet dataset
Please download the dataset from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`. Then, unzip using `unzip tiny-imagenet-200.zip` (make sure that the dataset is at `./datasets/tiny_imagenet_200`). Then, copy `./datasets/tinyimagenet_dataset_reorg.py` to `./datasets/tiny_imagenet_200` and run `cd ./datasets/tiny_imagenet_200; python tinyimagenet_dataset_reorg.py`. This ensures that the dataset is in the correct format. 