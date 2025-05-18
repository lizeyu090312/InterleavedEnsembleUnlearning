'''
This is the implement of BadNets [1].

Reference:
[1] Badnets: Evaluating Backdooring Attacks on Deep Neural Networks. IEEE Access 2019.
'''

import copy
import random

import numpy as np
import PIL
from PIL import Image, ImageChops
import torchvision.transforms as transforms
from torchvision.datasets import GTSRB
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, RandomAffine, ToTensor, ToPILImage
from torchvision.utils import save_image
from .base import *

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

class PoisonedTrainDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 poisoned_transform_index,
                 poisoned_target_transform_index, 
                 shuffled_indices):
        super(PoisonedTrainDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        total_num = len(benign_dataset)
        assert total_num == len(shuffled_indices)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        self.poisoned_set = frozenset(shuffled_indices[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = Compose([copy.deepcopy(self.transform)])

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        transform1 = Compose([RandomAffine(degrees=10)])
        # transform2 = Compose([ToTensor()])
        # transform3 = Compose([ToPILImage()])
        if index in self.poisoned_set:
            sample = sample.rotate(16)
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = transform1(sample)
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target

class PoisonedTestDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 poisoned_transform_index,
                 poisoned_target_transform_index, 
                 shuffled_indices):
        super(PoisonedTestDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        total_num = len(benign_dataset)
        assert total_num == len(shuffled_indices)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        self.poisoned_set = frozenset(shuffled_indices[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        transform1 = Compose([RandomAffine(degrees=10)])
        # transform2 = Compose([ToTensor()])
        # transform3 = Compose([ToPILImage()])
        if index in self.poisoned_set:
            sample = sample.rotate(16)
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = transform1(sample)
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target

class PoisonedTrainMNIST(MNIST):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 poisoned_transform_index,
                 poisoned_target_transform_index, 
                 shuffled_indices):
        super(PoisonedTrainMNIST, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        assert total_num == len(shuffled_indices)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        self.poisoned_set = frozenset(shuffled_indices[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
    
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        transform1 = Compose([RandomAffine(degrees=10)])
        transform2 = Compose([ToTensor()])
        transform3 = Compose([ToPILImage()])
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            img = transform3(img)
            img = img.rotate(16)
            img = transform2(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)
                img = transform1(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

class PoisonedTestMNIST(MNIST):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 poisoned_transform_index,
                 poisoned_target_transform_index, 
                 shuffled_indices):
        super(PoisonedTestMNIST, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        assert total_num == len(shuffled_indices)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        self.poisoned_set = frozenset(shuffled_indices[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        transform1 = Compose([RandomAffine(degrees=10)])
        transform2 = Compose([ToTensor()])
        transform3 = Compose([ToPILImage()])
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            img = transform3(img)
            img = img.rotate(16)
            img = transform2(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)
                img = transform1(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedTrainCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 shuffled_indices):
        super(PoisonedTrainCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=False)
        total_num = len(benign_dataset)
        assert total_num == len(shuffled_indices)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        self.poisoned_set = frozenset(shuffled_indices[:poisoned_num])


        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        self.transform1 = Compose([RandomAffine(degrees=10)])
        self.transform2 = Compose([ToTensor()])
        self.transform3 = Compose([ToPILImage()])
    
    def poison_transf(self, i):
        i = self.transform3(i)
        i = i.rotate(16)
        return self.transform2(i)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        """
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            if type(img) == tuple:  # GMML transform
                a, b, c, d = img
                img = [self.poison_transf(a[idxx]) for idxx in range(len(a))], [self.poison_transf(b[idxx]) for idxx in range(len(b))], \
                    [self.poison_transf(c[idxx]) for idxx in range(len(c))], [self.poison_transf(d[idxx])  for idxx in range(len(d))]
            else:  # basic simple transform
                img = self.transform3(img)
                img = img.rotate(16)
                img = self.transform2(img)
        else:
            if self.transform is not None:
                if type(img) == tuple:
                    a, b, c, d = img
                    img = [self.transform1(self.transform(a[idxx])) for idxx in range(len(a))], [self.transform1(self.transform(b[idxx])) for idxx in range(len(b))], \
                    [self.transform1(self.transform(c[idxx])) for idxx in range(len(c))], [self.transform1(self.transform(d[idxx])) for idxx in range(len(d))]
                else:
                    img = self.transform(img)
                    img = self.transform1(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
        """
        if index in self.poisoned_set:
            # img = self.poisoned_transform(img)
            # img = transform3(img)
            # img = img.rotate(16)
            # img = transform2(img)
            img = img.rotate(16)
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform1(img)
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedTrainGTSRB(GTSRB):
    def __init__(self,
                 benign_dataset:GTSRB,
                 y_target,
                 poisoned_rate,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 shuffled_indices):
        super(PoisonedTrainGTSRB, self).__init__(
            benign_dataset.root,
            benign_dataset._split,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=False)
        total_num = len(benign_dataset)
        assert total_num == len(shuffled_indices)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        self.poisoned_set = frozenset(shuffled_indices[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        self.resize32 = transforms.Resize((32, 32))

        self.transform1 = Compose([RandomAffine(degrees=10)])
        self.transform2 = Compose([ToTensor()])
        self.transform3 = Compose([ToPILImage()])
    
    def poison_transf(self, i):
        i = self.transform3(i)
        i = i.rotate(16)
        return self.transform2(i)

    def __getitem__(self, index):
        img_path, target = self._samples[index]
        target = int(target)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.resize32(Image.open(img_path).convert("RGB"))
        """
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            if type(img) == tuple:  # GMML transform
                a, b, c, d = img
                img = [self.poison_transf(a[idxx]) for idxx in range(len(a))], [self.poison_transf(b[idxx]) for idxx in range(len(b))], \
                    [self.poison_transf(c[idxx]) for idxx in range(len(c))], [self.poison_transf(d[idxx])  for idxx in range(len(d))]
            else:  # basic simple transform
                img = self.transform3(img)
                img = img.rotate(16)
                img = self.transform2(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)
                if type(img) == tuple:
                    a, b, c, d = img
                    img = [self.transform1(a[idxx]) for idxx in range(len(a))], [self.transform1(b[idxx]) for idxx in range(len(b))], \
                    [self.transform1(c[idxx]) for idxx in range(len(c))], [self.transform1(d[idxx]) for idxx in range(len(d))]
                else:
                    img = self.transform1(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        """
        if index in self.poisoned_set:
            # img = self.poisoned_transform(img)
            # img = transform3(img)
            # img = img.rotate(16)
            # img = transform2(img)
            img = img.rotate(16)
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform1(img)
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedTestCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 poisoned_transform_index,
                 poisoned_target_transform_index, 
                 shuffled_indices):
        super(PoisonedTestCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=False)
        total_num = len(benign_dataset)
        assert total_num == len(shuffled_indices)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        self.poisoned_set = frozenset(shuffled_indices[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        transform1 = Compose([RandomAffine(degrees=10)])
        transform2 = Compose([ToTensor()])
        transform3 = Compose([ToPILImage()])

        if index in self.poisoned_set:
            # img = self.poisoned_transform(img)
            # img = transform3(img)
            # img = img.rotate(16)
            # img = transform2(img)
            img = img.rotate(16)
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = transform1(img)
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


def CreatePoisonedTrainDataset(benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index):
    class_name = type(benign_dataset)
    if class_name == DatasetFolder:
        return PoisonedTrainDatasetFolder(benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == MNIST:
        return PoisonedTrainMNIST(benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == CIFAR10:
        return PoisonedTrainCIFAR10(benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index)
    else:
        raise NotImplementedError

def CreatePoisonedTestDataset(benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index):
    class_name = type(benign_dataset)
    if class_name == DatasetFolder:
        return PoisonedTestDatasetFolder(benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == MNIST:
        return PoisonedTestMNIST(benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == CIFAR10:
        return PoisonedTestCIFAR10(benign_dataset, y_target, poisoned_rate, poisoned_transform_index, poisoned_target_transform_index)
    else:
        raise NotImplementedError


class BATT(Base):
    """Construct poisoned datasets with BadNets method.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        poisoned_transform_train_index (int): The position index that poisoned transform will be inserted in train dataset. Default: 0.
        poisoned_transform_test_index (int): The position index that poisoned transform will be inserted in test dataset. Default: 0.
        poisoned_target_transform_index (int): The position that poisoned target transform will be inserted. Default: 0.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 schedule=None,
                 seed=0,
                 deterministic=False,
                 ):

        super(BATT, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.poisoned_train_dataset = CreatePoisonedTrainDataset(
            train_dataset,
            y_target,
            poisoned_rate,
            poisoned_transform_train_index,
            poisoned_target_transform_index)

        self.poisoned_test_dataset = CreatePoisonedTestDataset(
            test_dataset,
            y_target,
            1,
            poisoned_transform_test_index,
            poisoned_target_transform_index)
