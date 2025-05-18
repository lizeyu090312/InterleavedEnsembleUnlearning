import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import numpy as np
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST, GTSRB
import core

import sys, tqdm

from PIL import Image

class GetPoisonedDataset(torch.utils.data.Dataset):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = torch.FloatTensor(self.data_list[index])
        label = torch.FloatTensor(self.labels[index])
        return img, label


# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# Prepare datasets and follow the default data augmentation in the original paper


cifar_root = "./datasets/cifar10"
gtsrb_root = "./datasets"
tiny_imagenet_200_train_root = "./datasets/tiny_imagenet_200/train"
tiny_imagenet_200_test_root = "./datasets/tiny_imagenet_200/val/images"

if __name__ == "__main__":
    which_device = int(sys.argv[1])
    which_dataset = str(sys.argv[2])  # "cifar10", "gtsrb", "tiny_imagenet_200"

    transform_train = Compose([
        transforms.Resize((32, 32)) if which_dataset != "tiny_imagenet_200" else transforms.Resize((64, 64)),
        RandomHorizontalFlip(),
        ToTensor(),
    ])
    transform_test = Compose([
        transforms.Resize((32, 32)) if which_dataset != "tiny_imagenet_200" else transforms.Resize((64, 64)),
        ToTensor(),
    ])

    def loader(img):
        return Image.open(img).convert("RGB")
    
    if which_dataset == "cifar10":
        trainset = CIFAR10(root=cifar_root, transform=transform_train, target_transform=None, train=True)
        testset = CIFAR10(root=cifar_root, transform=transform_test, target_transform=None, train=False)
    elif which_dataset == "gtsrb":
        trainset = GTSRB(root=gtsrb_root, transform=transform_train, target_transform=None, split='train')
        testset = GTSRB(root=gtsrb_root, transform=transform_test, target_transform=None, split='test')
    elif which_dataset == "tiny_imagenet_200":
        trainset = DatasetFolder(tiny_imagenet_200_train_root, transform=transform_train, loader=loader, 
                                 extensions='jpeg')
        testset  = DatasetFolder(tiny_imagenet_200_test_root, transform=transform_test, loader=loader, 
                                 extensions='jpeg')
    secret_size = 20

    train_data_set = []
    train_secret_set = []
    for idx, (img, lab) in tqdm.tqdm(enumerate(trainset)):
        train_data_set.append(img.tolist())
        secret = np.random.binomial(1, .5, secret_size).tolist()
        train_secret_set.append(secret)


    for idx, (img, lab) in tqdm.tqdm(enumerate(testset)):
        train_data_set.append(img.tolist())
        secret = np.random.binomial(1, .5, secret_size).tolist()
        train_secret_set.append(secret)
    print("Done with enumerating through datasets")

    train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)


    schedule = {'device': 'GPU', #'CUDA_VISIBLE_DEVICES': '1', 
                'GPU_num': 1,

        'benign_training': False, 'batch_size': 128, 'num_workers': 2,

        'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4, 'gamma': 0.1, 'schedule': [150, 180],

        'epochs': 200,

        'log_iteration_interval': 100, 'test_epoch_interval': 10, 'save_epoch_interval': 100,

        'save_dir': 'experiments', 
        'experiment_name': 'train_poison_DataFolder_CIFAR10_ISSBA' if which_dataset == "cifar10" else f'train_poison_DataFolder_{which_dataset}_ISSBA'
    }

    encoder_schedule = {
        'secret_size': secret_size,
        'enc_height': 32 if which_dataset != "tiny_imagenet_200" else 64,
        'enc_width': 32 if which_dataset != "tiny_imagenet_200" else 64,
        'enc_in_channel': 3,
        'enc_total_epoch': 20,
        'enc_secret_only_epoch': 2,
        'enc_use_dis': False,
    }

    # Configure the attack scheme
    ISSBA = core.ISSBA(dataset_name="cifar10" if which_dataset == "cifar10" else "gtsrb", train_dataset=trainset, test_dataset=testset, train_steg_set=train_steg_set,
        model=core.models.ResNet(18), loss=nn.CrossEntropyLoss(), y_target=1, poisoned_rate=0.1,
        encoder_schedule=encoder_schedule, encoder=None, schedule=schedule, seed=0, deterministic=False)

    ISSBA.train(schedule=schedule, which_device=which_device, train_encoder_decoder_only=True)
