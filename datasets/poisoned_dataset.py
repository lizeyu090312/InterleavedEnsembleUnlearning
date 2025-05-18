import random
import os
from typing import Callable, Optional
import typing
from PIL import Image

from torchvision.datasets import CIFAR10, GTSRB, DatasetFolder
from torchvision import transforms
import torch

import numpy as np

from .BackdoorBox import core
import cv2
from PIL import Image
from utils import loader


class PoisonISSBA:
    def __init__(self, trigger_label, poisoning_rate, dataset, dataset_name, shuffled_indices, train: bool = True, 
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__()
        hw = 64 if dataset_name == 'tiny_imagenet_200' else 32
        encoder_schedule = {'secret_size': 20, 'enc_height': hw, 'enc_width': hw, 'enc_in_channel': 3, 'enc_total_epoch': 20,
            'enc_secret_only_epoch': 2, 'enc_use_dis': False}
        issba_obj = core.ISSBA(dataset_name=dataset_name, train_dataset=dataset, test_dataset=None, train_steg_set=None,
            model=None, loss=None, y_target=trigger_label, poisoned_rate=poisoning_rate if train else 1.0,
            encoder_schedule=None, encoder=None, schedule=None, seed=0, deterministic=False, shuffled_indices=shuffled_indices)
        self.transform = transform
        self.poisoned_set = issba_obj.poisoned_set
        self.target_transform = target_transform
        checkpoints_base = "./checkpoints/"
        if dataset_name == "CIFAR10":
            chkp_path = os.path.join(checkpoints_base, "CIFAR10_ISSBA_encoder_2024-07-15_16:37:57/encoder_decoder.pth")
        elif dataset_name == "gtsrb":
            chkp_path = os.path.join(checkpoints_base, "GTSRB_ISSBA_2024-07-24_20:19:59/encoder_decoder.pth")
        elif dataset_name == "tiny_imagenet_200":
            chkp_path = os.path.join(checkpoints_base, "tiny_imagenet_200_ISSBA_2024-07-31_05:02:56/encoder_decoder.pth")
        
        self.concated_dataset = issba_obj.get_dataset(
            dataset, device='cuda:0', chkp_path=chkp_path, encoder_schedule=encoder_schedule)
        print(f"Poison {len(issba_obj.poisoned_set)} over {len(self.concated_dataset)} samples (poisoning rate {issba_obj.poisoned_rate})")

    def __len__(self) -> int:
        return len(self.concated_dataset)
    
    def __getitem__(self, index):
        img, target = self.concated_dataset[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target.type(torch.LongTensor)


class CIFAR10Trojan(CIFAR10):
    def __init__(self, trigger_label, poisoning_rate, root: str, pattern, shuffled_indices, train: bool = True,
                 transform = None, target_transform = None, 
                 download: bool = False) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.poisoning_rate = poisoning_rate if train else 1.0
        self.trigger_img = np.array(pattern, dtype=np.float32)
        self.poisoned_set = set(shuffled_indices[:int(len(self.targets) * poisoning_rate)])
        self.trigger_label = trigger_label

    def put_trigger(self, img):
        img = img + self.trigger_img
        img[img >= 255] = 255
        img[img <= 0] = 0  # clipping
        return Image.fromarray(np.array(img, dtype=np.uint8))
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poisoned_set:
            target = self.trigger_label
            img = self.put_trigger(img)

        if self.transform is not None:
            img = self.transform(Image.fromarray(np.array(img, dtype=np.uint8)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class GTSRBTrojan(GTSRB):
    def __init__(self, trigger_label, poisoning_rate, root: str, pattern, shuffled_indices, train: bool = True,
                 transform = None, target_transform = None, 
                 download: bool = False) -> None:
        super().__init__(root, split="train" if train==True else "test", transform=transform, target_transform=target_transform, download=download)
        self.poisoning_rate = poisoning_rate if train else 1.0
        self.trigger_img = np.array(pattern, dtype=np.float32)
        self.poisoned_set = set(shuffled_indices[:int(len(self._samples) * poisoning_rate)])
        self.trigger_label = trigger_label
        self.resize32 = transforms.Resize((32, 32))

    def put_trigger(self, img):
        img = img + self.trigger_img
        img[img >= 255] = 255
        img[img <= 0] = 0  # clipping
        return Image.fromarray(np.array(img, dtype=np.uint8))
    
    def __getitem__(self, index):
        img_path, target = self._samples[index]
        target = int(target)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.asarray(self.resize32(Image.open(img_path).convert("RGB")))
        if index in self.poisoned_set:
            target = self.trigger_label
            img = self.put_trigger(img)

        if self.transform is not None:
            img = self.transform(Image.fromarray(np.array(img, dtype=np.uint8)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MiniImageNetTrojan(DatasetFolder):
    def __init__(self, trigger_label, poisoning_rate, root: str, pattern, extensions, loader, shuffled_indices,
                  train: bool = True, transform = None, target_transform = None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform, 
                         extensions=extensions, loader=loader)
        self.poisoning_rate = poisoning_rate if train else 1.0
        self.trigger_img = np.array(pattern, dtype=np.float32)
        self.poisoned_set = set(shuffled_indices[:int(len(self.targets) * poisoning_rate)])
        self.trigger_label = trigger_label

    def put_trigger(self, img):
        img = img + self.trigger_img
        img[img >= 255] = 255
        img[img <= 0] = 0  # clipping
        return Image.fromarray(np.array(img, dtype=np.uint8))
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poisoned_set:
            target = self.trigger_label
            img = self.put_trigger(img)

        if self.transform is not None:
            img = self.transform(Image.fromarray(np.array(img, dtype=np.uint8)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MiniImageNet_L0(MiniImageNetTrojan):
    def __init__(self, trigger_label, poisoning_rate, root: str, pattern, extensions, loader, mask, 
                 shuffled_indices, train: bool = True, transform = None, target_transform = None) -> None:
        
        MiniImageNetTrojan.__init__(
            self, trigger_label, poisoning_rate, root, pattern, extensions, loader, shuffled_indices, train, transform, target_transform)
        self.l0_mask = mask

    def put_trigger(self, img):
        img = img*self.l0_mask + self.trigger_img
        img[img >= 255] = 255
        img[img <= 0] = 0  # clipping
        return Image.fromarray(np.array(img, dtype=np.uint8))


class CIFAR10_L0(CIFAR10Trojan):
    def __init__(self, trigger_label, poisoning_rate, root: str, pattern, mask, 
                 shuffled_indices, train: bool = True, transform = None, target_transform = None, 
                 download: bool = False) -> None:
        CIFAR10Trojan.__init__(
            self, trigger_label, poisoning_rate, root, pattern, shuffled_indices, train, transform, target_transform, download)
        self.l0_mask = mask

    def put_trigger(self, img):
        img = img*self.l0_mask + self.trigger_img
        img[img >= 255] = 255
        img[img <= 0] = 0  # clipping
        return Image.fromarray(np.array(img, dtype=np.uint8))


class GTSRB_L0(GTSRBTrojan):
    def __init__(self, trigger_label, poisoning_rate, root: str, pattern, mask, 
                 shuffled_indices, train: bool = True, transform = None, target_transform = None, 
                 download: bool = False) -> None:
        GTSRBTrojan.__init__(
            self, trigger_label, poisoning_rate, root, pattern, shuffled_indices, train, transform, target_transform, download)
        self.l0_mask = mask

    def put_trigger(self, img):
        img = img*self.l0_mask + self.trigger_img
        img[img >= 255] = 255
        img[img <= 0] = 0  # clipping
        return Image.fromarray(np.array(img, dtype=np.uint8))


def get_poisoned_dataset(method, dataset_name:str, train:bool, transform, trigger_label, poisoning_rate):
    
    def read_image(img_path, type=None):
        img = cv2.imread(img_path)
        if type is None:
            return img
        elif isinstance(type,str) and type.upper() == "RGB":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(type,str) and type.upper() == "GRAY":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise NotImplementedError
    
    cifar_root = "./datasets/cifar10"
    gtsrb_root = "./datasets"
    imagenet_200_root = "./datasets/tiny_imagenet_200/train" if train else "./datasets/tiny_imagenet_200/val/images"
    to_tensor_resize32 = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])  # ToTensor normalizes to 0.0-1.0
    to_tensor_resize64 = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
    assert dataset_name in {"CIFAR10", "gtsrb", "tiny_imagenet_200"}
    if dataset_name == "CIFAR10":
        benign_dataset = CIFAR10(root=cifar_root, train=train,
                                   transform=to_tensor_resize32 if method == "ISSBA" else transform, 
                                   target_transform=None)
    elif dataset_name == "gtsrb":
        benign_dataset = GTSRB(root=gtsrb_root, split="train" if train == True else "test",
                                   transform=to_tensor_resize32 if method == "ISSBA" else transform, 
                                   target_transform=None)
    elif dataset_name == "tiny_imagenet_200":
        benign_dataset = DatasetFolder(imagenet_200_root, transform=to_tensor_resize64 if method == "ISSBA" else transform, 
                                       extensions="jpeg", loader=loader)
    poisoning_rate = poisoning_rate if train else 1.0
    shuffle_indices_dir = "./shuffle_indices"
    shuffled_indices = np.load(os.path.join(shuffle_indices_dir, f"{dataset_name}_shuffle_indices_train_{train}.npy"))
    if method == "BadNets-white":
        if dataset_name != "tiny_imagenet_200":
            pattern = torch.zeros((32, 32), dtype=torch.uint8)
            pattern[-3:, -3:] = 255

            weight = torch.zeros((32, 32), dtype=torch.float32)
            weight[-3:, -3:] = 1.0
            ds_class = core.BadNetsPoisonedCIFAR10 if dataset_name == "CIFAR10" else core.BadNetsPoisonedGTSRB
            ds = ds_class(
                benign_dataset=benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate,
                pattern=pattern, weight=weight, poisoned_transform_index=0, poisoned_target_transform_index=0, 
                shuffled_indices=shuffled_indices)
        else:
            pattern = torch.zeros((64, 64), dtype=torch.uint8)
            pattern[-6:, -6:] = 255
            weight = torch.zeros((64, 64), dtype=torch.float32)
            weight[-6:, -6:] = 1.0
            ds = core.BadNetsPoisonedDatasetFolder(
                benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate, pattern=pattern, 
                weight=weight, poisoned_transform_index=0, poisoned_target_transform_index=0, 
                shuffled_indices=shuffled_indices)
    elif method == "BadNets-pattern":
        if dataset_name != "tiny_imagenet_200":
            pattern = torch.zeros((3, 32, 32), dtype=torch.uint8)
            width = 32
            pattern[:, width - 1, width - 1] = 255
            pattern[:, width - 1, width - 2] = 0
            pattern[:, width - 1, width - 3] = 255

            pattern[:, width - 2, width - 1] = 0
            pattern[:, width - 2, width - 2] = 255
            pattern[:, width - 2, width - 3] = 0

            pattern[:, width - 3, width - 1] = 255
            pattern[:, width - 3, width - 2] = 0
            pattern[:, width - 3, width - 3] = 0

            weight = torch.zeros((32, 32), dtype=torch.float32)
            weight[-3:, -3:] = 1.0
            ds_class = core.BadNetsPoisonedCIFAR10 if dataset_name == "CIFAR10" else core.BadNetsPoisonedGTSRB
            ds = ds_class(
                benign_dataset=benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate,
                pattern=pattern, weight=weight, poisoned_transform_index=0, poisoned_target_transform_index=0, 
                shuffled_indices=shuffled_indices)
        else:
            pattern = torch.zeros((3, 64, 64), dtype=torch.uint8)
            width = 64
            pattern[:, width - 1, width - 1] = pattern[:, width - 1, width - 2] = 255
            pattern[:, width - 1, width - 3] = pattern[:, width - 1, width - 4] = 0
            pattern[:, width - 1, width - 5] = pattern[:, width - 1, width - 6] = 255

            pattern[:, width - 2, width - 1] = pattern[:, width - 2, width - 2] = 0
            pattern[:, width - 2, width - 3] = pattern[:, width - 2, width - 4] = 255
            pattern[:, width - 2, width - 5] = pattern[:, width - 2, width - 6] = 0

            pattern[:, width - 3, width - 1] = pattern[:, width - 3, width - 2] = 255
            pattern[:, width - 3, width - 3] = pattern[:, width - 3, width - 4] = 0
            pattern[:, width - 3, width - 5] = pattern[:, width - 3, width - 6] = 0

            weight = torch.zeros((64, 64), dtype=torch.float32)
            weight[-6:, -6:] = 1.0
            ds = core.BadNetsPoisonedDatasetFolder(
                benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate, pattern=pattern, 
                weight=weight, poisoned_transform_index=0, poisoned_target_transform_index=0, 
                shuffled_indices=shuffled_indices)
    elif method == "ISSBA":
        # note: transform must NOT include ToTensor()
        ds = PoisonISSBA(trigger_label, poisoning_rate, benign_dataset, dataset_name, 
                         shuffled_indices, train, transform=transform, target_transform=None)
    elif method == "BATT":
        if dataset_name != "tiny_imagenet_200":
            ds_class = core.BATTPoisonedCIFAR10 if dataset_name == "CIFAR10" else core.BATTPoisonedGTSRB
            ds = ds_class(
                benign_dataset=benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate,
                poisoned_transform_index=None, poisoned_target_transform_index=0, shuffled_indices=shuffled_indices)
        else:
            ds = core.BATTPoisonedDatasetFolder(
                benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate, 
                poisoned_transform_index=None, poisoned_target_transform_index=0, shuffled_indices=shuffled_indices)
    elif method == "Blended":
        if dataset_name != "tiny_imagenet_200":
            hello_kitty_img = to_tensor_resize32(read_image("./triggers/Blended_hellow_kitty.png"))*255
            ds_class = core.BlendedPoisonedCIFAR10 if dataset_name == "CIFAR10" else core.BlendedPoisonedGTSRB
            ds = ds_class(
                benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate, 
                poisoned_transform_index=0, pattern=hello_kitty_img, weight=0.4, poisoned_target_transform_index=0, 
                shuffled_indices=shuffled_indices)
        else:
            hello_kitty_img = to_tensor_resize64(read_image("./triggers/Blended_hellow_kitty.png"))*255
            ds = core.BlendedPoisonedDatasetFolder(
                benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate, 
                poisoned_transform_index=0, pattern=hello_kitty_img, weight=0.4, poisoned_target_transform_index=0, 
                shuffled_indices=shuffled_indices)
    elif method == "Trojan-WM":
        if dataset_name != "tiny_imagenet_200":
            trojan_watermark = cv2.resize(read_image("./triggers/Trojan-WM.jpg"), dsize=(32, 32))
            ds_class = CIFAR10Trojan if dataset_name == "CIFAR10" else GTSRBTrojan
            ds = ds_class(
                root=cifar_root if dataset_name == "CIFAR10" else gtsrb_root, trigger_label=trigger_label, poisoning_rate=poisoning_rate, 
                pattern=trojan_watermark, train=train, transform=transform, shuffled_indices=shuffled_indices)
        else:
            trojan_watermark = cv2.resize(read_image("./triggers/Trojan-WM.jpg"), dsize=(64, 64))
            ds = MiniImageNetTrojan(
                root=imagenet_200_root, trigger_label=trigger_label, poisoning_rate=poisoning_rate, pattern=trojan_watermark, 
                train=train, transform=transform, loader=loader, extensions='jpeg', shuffled_indices=shuffled_indices)
    elif method == "Trojan-SQ":
        if dataset_name != "tiny_imagenet_200":
            trojan_watermark = cv2.resize(read_image("./triggers/Trojan-SQ.jpg"), dsize=(32, 32))
            ds_class = CIFAR10Trojan if dataset_name == "CIFAR10" else GTSRBTrojan
            ds = ds_class(
                root=cifar_root if dataset_name == "CIFAR10" else gtsrb_root, trigger_label=trigger_label, poisoning_rate=poisoning_rate, 
                pattern=trojan_watermark, train=train, transform=transform, shuffled_indices=shuffled_indices)
        else:
            trojan_watermark = cv2.resize(read_image("./triggers/Trojan-SQ.jpg"), dsize=(64, 64))
            ds = MiniImageNetTrojan(
                root=imagenet_200_root, trigger_label=trigger_label, poisoning_rate=poisoning_rate, pattern=trojan_watermark, 
                train=train, transform=transform, loader=loader, extensions='jpeg', shuffled_indices=shuffled_indices)
    elif method == "Smooth":
        if dataset_name != "tiny_imagenet_200":
            smooth_watermark = np.array(np.load("./triggers/Smooth.npy")[0], dtype=np.float32)
            smooth_watermark *= 255
            ds_class = CIFAR10Trojan if dataset_name == "CIFAR10" else GTSRBTrojan
            ds = ds_class(
                root=cifar_root if dataset_name == "CIFAR10" else gtsrb_root, trigger_label=trigger_label, poisoning_rate=poisoning_rate, 
                pattern=smooth_watermark, train=train, transform=transform, shuffled_indices=shuffled_indices)
        else:
            smooth_watermark = cv2.resize(
                np.array(np.load("./triggers/Smooth.npy")[0], dtype=np.float32), dsize=(64, 64))
            smooth_watermark *= 255
            smooth_watermark[smooth_watermark >= 255] = 255
            smooth_watermark[smooth_watermark <= 0] = 0
            ds = MiniImageNetTrojan(
                root=imagenet_200_root, trigger_label=trigger_label, poisoning_rate=poisoning_rate, pattern=smooth_watermark, 
                train=train, transform=transform, loader=loader, extensions='jpeg', shuffled_indices=shuffled_indices)
    elif method == "l0-inv":
        if dataset_name != "tiny_imagenet_200":
            l0_img = np.array(read_image("./triggers/l0_inv.png"), dtype=np.float32)
            mask = 1 - np.transpose(np.load("./triggers/l0_inv_mask.npy"), (1, 2, 0))
            ds_class = CIFAR10_L0 if dataset_name == "CIFAR10" else GTSRB_L0
            ds = ds_class(
                root=cifar_root if dataset_name == "CIFAR10" else gtsrb_root, trigger_label=trigger_label, poisoning_rate=poisoning_rate, 
                pattern=l0_img, mask=mask, train=train, transform=transform, shuffled_indices=shuffled_indices)
        else:
            l0_img = cv2.resize(
                np.array(read_image("./triggers/l0_inv.png"), dtype=np.float32), dsize=(64, 64))
            mask = 1 - cv2.resize(np.transpose(np.load("./triggers/l0_inv_mask.npy"), (1, 2, 0)), dsize=(64, 64))
            ds = MiniImageNet_L0(
                root=imagenet_200_root, trigger_label=trigger_label, poisoning_rate=poisoning_rate, pattern=l0_img, 
                train=train, transform=transform, mask=mask, loader=loader, extensions='jpeg', shuffled_indices=shuffled_indices)
    elif method == "l2-inv":
        if dataset_name != "tiny_imagenet_200":
            l2_img = np.array(read_image("./triggers/l2_inv.png"), dtype=np.float32)
            ds_class = CIFAR10Trojan if dataset_name == "CIFAR10" else GTSRBTrojan
            ds = ds_class(
                root=cifar_root if dataset_name == "CIFAR10" else gtsrb_root, trigger_label=trigger_label, poisoning_rate=poisoning_rate, 
                pattern=l2_img, train=train, transform=transform, shuffled_indices=shuffled_indices)
        else:
            l2_img = cv2.resize(
                np.array(read_image("./triggers/l2_inv.png"), dtype=np.float32), dsize=(64, 64))
            ds = MiniImageNetTrojan(
                root=imagenet_200_root, trigger_label=trigger_label, poisoning_rate=poisoning_rate, pattern=l2_img, 
                train=train, transform=transform, loader=loader, extensions='jpeg', shuffled_indices=shuffled_indices)
    elif method == "SIG":
        if dataset_name != "tiny_imagenet_200":
            sig_img = to_tensor_resize32(np.expand_dims(np.load("./triggers/SIG.npy"), -1))*255
            ds_class = core.BlendedPoisonedCIFAR10 if dataset_name == "CIFAR10" else core.BlendedPoisonedGTSRB
            ds = ds_class(
                benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate, 
                poisoned_transform_index=0, pattern=sig_img, weight=0.2, poisoned_target_transform_index=0, 
                shuffled_indices=shuffled_indices, is_clean_label=True, training=train)
        else:
            sig_img = to_tensor_resize64(np.expand_dims(np.load("./triggers/SIG.npy"), -1))*255
            ds = core.BlendedPoisonedDatasetFolder(
                benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate, 
                poisoned_transform_index=0, pattern=sig_img, weight=0.2, poisoned_target_transform_index=0, 
                shuffled_indices=shuffled_indices, is_clean_label=True, training=train)
        if train == True:
            ds.poisoned_target_transform = transforms.Compose([])  # clean label == don't change label during training
    elif method == "WaNet":
        def gen_grid_WaNet(height, k):
            """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
            according to the input height ``height`` and the uniform grid size ``k``.
            """
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
            noise_grid = torch.nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
            noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
            array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
            x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
            identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

            return identity_grid, noise_grid

        assert dataset_name == "CIFAR10" or dataset_name == "GTSRB"
        identity_grid, noise_grid = gen_grid_WaNet(32, 4)
        ds_class = core.WaNetPoisonedCIFAR10 if dataset_name == "CIFAR10" else core.WaNetPoisonedGTSRB
        ds = ds_class(
            benign_dataset, y_target=trigger_label, poisoned_rate=poisoning_rate, identity_grid=identity_grid, 
            noise_grid=noise_grid, noise=False, poisoned_transform_index=0, poisoned_target_transform_index=0)
    else:
        raise NotImplementedError
    return ds


class PoisonedData:
    """This method should queue images that are flagged by poison_module as 'poisoned' 
       and allows robust_module to consume queued samples"""
    def __init__(self, batch_size, device, present_keys):
        super().__init__()
        self.clean_crops_tens, self.corrupted_crops_tens, self.no_crop_imgs_tens, self.labels_tens = [torch.zeros(0, device=device) for _ in range(4)]
        self.batch_size = batch_size
        self.curr_len = 0
        self.present_keys = present_keys

    def _validate_inp(self, inp:dict[str, torch.Tensor]) -> None:
        assert self.present_keys == set(inp.keys()), f"inp doesn't contain the correct keys, inp.keys()={inp.keys()}; should contain {self.present_keys}"
        dim0_len = inp["clean_crops"].shape[0]
        for k, v in inp.items():
            err_str = f"in _validate_inp: k={k}, v.shape={v.shape}, dim0_len={dim0_len}"
            assert int(v.shape[0]) == int(dim0_len), err_str
            right_ndim = 4 if k != "labels" else 1
            assert int(v.ndim) == right_ndim, err_str + str(f" v.ndim={v.ndim}, right_ndim={right_ndim}")
        return
    
    def add_sample(self, inp:dict[str, torch.Tensor]) -> typing.Tuple[torch.Tensor] | bool: 
        '''expects ["clean_crops": tensor with ndims = 4, (this is not compulsory =>) and assumes inp_len < self.batch_size '''
        self._validate_inp(inp)

        inp_len = inp["clean_crops"].shape[0]
        amount_to_concat = min(inp_len, self.batch_size - self.curr_len)
        if self.batch_size - self.curr_len <= 0:  # this should never happen. 
            return self._reinit_and_return(inp)
        # no need to worry about amount_to_concat=0 since concatenate handles this
        self.clean_crops_tens = torch.concatenate((self.clean_crops_tens, inp["clean_crops"][0:amount_to_concat]), dim=0)
        self.corrupted_crops_tens = torch.concatenate((self.corrupted_crops_tens, inp["corrupted_crops"][0:amount_to_concat]), dim=0)
        self.no_crop_imgs_tens = torch.concatenate((self.no_crop_imgs_tens, inp["no_crop_imgs"][0:amount_to_concat]), dim=0)
        self.labels_tens = torch.concatenate((self.labels_tens, inp["labels"][0:amount_to_concat]), dim=0)
        self.curr_len += amount_to_concat

        if self.curr_len >= self.batch_size:
            leftover_inp = {k: v[amount_to_concat:] for k, v in inp.items()}
            return self._reinit_and_return(leftover_inp)
        return False, False, False, False  # doesn't have enough for a batch yet.
    
    def _reinit_and_return(self, inp:dict[str, torch.Tensor]) -> typing.Tuple[torch.Tensor]:
        to_ret = self.clean_crops_tens[0:self.batch_size], self.corrupted_crops_tens[0:self.batch_size], \
            self.no_crop_imgs_tens[0:self.batch_size], self.labels_tens[0:self.batch_size]
        self.clean_crops_tens, self.corrupted_crops_tens, self.no_crop_imgs_tens, self.labels_tens = \
            inp["clean_crops"], inp["corrupted_crops"], inp["no_crop_imgs"], inp["labels"]
        self.curr_len = self.clean_crops_tens.shape[0]
        return to_ret
