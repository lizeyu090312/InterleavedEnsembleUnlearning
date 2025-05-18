""" Should be used for data augmentation. See poisoned_dataset.py for backdoors """
from PIL import Image
import numpy as np
import random

from numpy.random import randint
import torch

from PIL import ImageFilter, ImageOps
from torchvision import transforms as tf

from torchvision import transforms


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def getItem(idx, X, target = None, transform=None, training_mode = 'SSL'):
    if transform is not None:
        X = transform(X)

    return X, target


class myRandCrop(tf.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super(myRandCrop, self).__init__(size, scale, ratio, interpolation)
        
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return tf.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i, j, h, w)


class myRandomHorizontalFlip(tf.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(myRandomHorizontalFlip, self).__init__(p=p)
        
    def forward(self, img):
        if torch.rand(1) < self.p:
            return tf.functional.hflip(img), 1
        return img, 0
    
    
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def GMML_replace_list(samples, corrup_prev, masks_prev, drop_type='noise', max_replace=0.35, align=16):
    # samples is from clean_crop, corrupt_prev is from corrupt_crop
    rep_drop = 1 if drop_type == '' else ( 1 / ( len(drop_type.split('-')) + 1 ) )  # rep_drop \in [0,1]
    
    n_imgs = samples.size()[0] #this is batch size, but in case bad inistance happened while loading
    samples_aug = samples.detach().clone()
    masks = torch.zeros_like(samples_aug)
    for i in range(n_imgs):
        idx_rnd = randint(0, n_imgs)
        if random.random() < rep_drop: # replaces some patches in clean crop (which is samples_aug) 
            # with patches from corrupt_crops in the same batch
            samples_aug[i], masks[i] = GMML_drop_rand_patches(samples_aug[i], corrup_prev[idx_rnd], max_replace=max_replace, align=align)
        else: # replaces clean crop (in samples_aug) with corrupted crop
            samples_aug[i], masks[i] = corrup_prev[i], masks_prev[i]

    return samples_aug, masks

def GMML_drop_rand_patches(X, X_rep=None, drop_type='noise', max_replace=0.7, align=16, max_block_sz=0.3):
    #######################
    # max_replace: percentage of image to be replaced
    # align: align corruption with the patch sizes
    # max_block_sz: percentage of the maximum block to be dropped
    #######################
   
    np.random.seed()    
    C, H, W = X.size()
    n_drop_pix = np.random.uniform(min(0.05, max_replace), max_replace)*H*W
    mx_blk_height = int(H*max_block_sz)
    mx_blk_width = int(W*max_block_sz)
    
    align = max(1, align)
    
    mask = torch.zeros_like(X)
    # drop_t = np.random.choice(drop_type.split('-'))
    
    while mask[0].sum() < n_drop_pix:
        drop_t = np.random.choice(drop_type.split('-'))
        ####### get a random block to replace 
        rnd_r = ( randint(0, H-align) // align ) * align
        rnd_c = ( randint(0, W-align) // align ) * align

        rnd_h = min(randint(align, mx_blk_height), H-rnd_r)
        rnd_h = round( rnd_h / align ) * align
        rnd_w = min(randint(align, mx_blk_width), W-rnd_c)
        rnd_w = round( rnd_w / align ) * align
        
        if X_rep is not None:
            X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = X_rep[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w].detach().clone()
        else:
            if drop_t == 'noise':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.empty((C, rnd_h, rnd_w), dtype=X.dtype, device=X.device).normal_()
            elif drop_t == 'zeros':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.zeros((C, rnd_h, rnd_w), dtype=X.dtype, device=X.device)
            else:
                ####### get a random block to replace from
                rnd_r2 = (randint(0, H-rnd_h) // align ) * align
                rnd_c2 = (randint(0, W-rnd_w) // align ) * align
            
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = X[:, rnd_r2:rnd_r2+rnd_h, rnd_c2:rnd_c2+rnd_w].detach().clone()
            
        mask[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = 1 
         
    return X, mask


class Noiser(object):
    def __init__(self, std_frac):
        super().__init__()
        self.std_frac = abs(float(std_frac))
        
    def __call__(self, x):
        if self.std_frac <= 1e-8:
            return x
        max_, min_ = torch.amax(x), torch.amin(x)
        return torch.clamp(torch.randn_like(x) * self.std_frac*(max_-min_) + x, min=min_, max=max_)


class DataAugmentationSiT(object):
    def __init__(self, args, add_ToPILImage:bool=False, visualise=False, noiser_std_perc:float=0.0):
        
        # for corruption
        self.drop_perc = args.drop_perc
        self.drop_type = args.drop_type
        self.drop_align = args.drop_align

        self.no_color_jitter = args.no_color_jitter
        self.noiser_obj = Noiser(noiser_std_perc)
        
        temp = [transforms.RandomHorizontalFlip(p=0.5)]
        if self.no_color_jitter == False:  # do not apply color jitter
            temp.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8))
            temp.append(transforms.RandomGrayscale(p=0.2))
        flip_and_color_jitter = transforms.Compose(temp)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR 10 (but in practice used for all datasets)
        ])
        self.add_ToPILImage = add_ToPILImage

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.75, 1.), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            # GaussianBlur(1.0),
            # self.normalize,
        ])
        self.resize = transforms.Resize(args.img_size, interpolation=Image.BICUBIC)
        self.to_pil = transforms.ToPILImage()
        self.visualise = visualise

    def __call__(self, image):
        clean_crops = []
        corrupted_crops = []
        masks_crops = []
        no_crop_imgs = []
        
        ## augmented 1
        im_no_crop = self.normalize(self.resize(image)) if self.add_ToPILImage == False else self.normalize(self.resize(self.to_pil(image)))
        im_no_crop = self.noiser_obj(im_no_crop)
        if self.visualise == True:
            im_no_crop = transforms.ToTensor()(self.resize(image)) if self.add_ToPILImage == False \
                else transforms.ToTensor()(self.resize(self.to_pil(image)))  
        # # ^^ for visualization purposes
        
        
        im_clean = self.normalize(self.global_transfo1(image)) if self.add_ToPILImage == False else self.normalize(self.global_transfo1(self.to_pil(image)))
        # im_orig no longer receives resized_crop, but im_corrupted is cropped using resized_crop
        im_corrupted = im_clean.detach().clone()
        im_mask = torch.zeros_like(im_corrupted)
        if self.drop_perc > 0:
            im_corrupted, im_mask = GMML_drop_rand_patches(im_corrupted, 
                                                           max_replace=self.drop_perc, drop_type=self.drop_type, align=self.drop_align)
        clean_crops.append(im_clean)
        corrupted_crops.append(im_corrupted)
        masks_crops.append(im_mask)
        no_crop_imgs.append(im_no_crop)

        return clean_crops, corrupted_crops, masks_crops, no_crop_imgs


def basic_transforms(args, train=False, no_color_jitter=False, add_ToPILImage=False):
    normalize_list = [
        transforms.Resize((args.img_size, args.img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR 10 (but in practice used for all datasets)
    ]
    if add_ToPILImage == True:
        normalize_list.insert(0, transforms.ToPILImage())
    normalize = transforms.Compose(normalize_list)
    if train:
        # basic transforms implemented in dataset_utils. Note: no cropping is allowed
        temp = [transforms.RandomHorizontalFlip(p=0.5)]
        if no_color_jitter == False:  # apply color jitter
            temp.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8))
            temp.append(transforms.RandomGrayscale(p=0.2))
        flip_and_color_jitter = transforms.Compose(temp)
        return transforms.Compose([flip_and_color_jitter, normalize])
    else:
        return normalize
