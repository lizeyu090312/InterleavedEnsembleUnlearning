import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from datasets import datasets_utils
from datasets.poisoned_dataset import get_poisoned_dataset

import utils

import torchvision
from train_utils import *

from cat_adaptive_attack_train_ensemble import CAT_AttackDataset


def train_teacher_SiT(args):
    # torch.autograd.set_detect_anomaly(True)

    with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    cudnn.benchmark = False
    cudnn.deterministic = True
    # preparing loss
    device = torch.device(f'cuda:{args.device}')
    loss_obj = nn.CrossEntropyLoss()

    # prepare dataset
    transform = datasets_utils.DataAugmentationSiT(args, add_ToPILImage=False if args.poison_method != "ISSBA" else True)
    transform.normalize = torchvision.transforms.Compose([])  # don't normalise since this was done during datagen
    transform_orig = datasets_utils.DataAugmentationSiT(args, add_ToPILImage=False if args.poison_method != "ISSBA" else True)
    if args.data_set == 'CIFAR10' or args.data_set == 'gtsrb' or args.data_set == 'tiny_imagenet_200':
        match args.data_set:
            case "CIFAR10":
                num_classes = 10
            case "gtsrb":
                num_classes = 43
            case "tiny_imagenet_200":
                num_classes = 200
        dataset = CAT_AttackDataset(root=f"checkpoints/adaptive_attack_backdoor_img_only/{args.data_set}/ensemble_{args.poison_method}/data/train_img", 
                                transform=transform, transform_orig=transform_orig, train=True, 
                                trigger_label=args.trigger_label, num_classes=num_classes, 
                                data_set=args.data_set)
    else:
        raise NotImplementedError("only supports CIFAR10 for poisoning for now")
    print("Successfully loaded in poisoned dataset")
    
    # building networks 
    teacher, _ = return_model(args, student_or_teacher="teacher", num_classes=num_classes)

    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, 
        collate_fn=collate_batch(args.drop_replace, args.drop_align, apply_poison=True))

    teacher = teacher.to(device)
    
    # preparing optimizer 
    optimizer = torch.optim.AdamW(utils.get_params_groups(teacher))  # to use with ViTs


    # init schedulers 
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256., 
        args.min_lr, args.finetuning_epochs, len(data_loader), warmup_epochs=args.warmup_epochs)
    
    wd_schedule = utils.cosine_scheduler( args.weight_decay,
        args.weight_decay_end, args.finetuning_epochs, len(data_loader))

    state_dict_teacher = torch.load(f"checkpoints/SiT_Small_ImageNet_ViT_student.pth", map_location=device)
    state_dict_teacher = remove_distr(state_dict_teacher)
    with (Path(args.output_dir) / "log.txt").open("a") as f:
        msg = teacher.load_state_dict(state_dict_teacher, strict=False)
        print(msg)
        f.write(f"Loading in SiT_Small_ImageNet_ViT_student.pth, msg={msg}\n")

    start_epoch = 0
    start_time = time.time()
    print("Training ..")
    for epoch in range(start_epoch, args.finetuning_epochs):
        
        # Training
        train_stats = train_one_epoch(teacher, loss_obj, 
            data_loader, optimizer, lr_schedule, wd_schedule, epoch, args)
        # logs
        teacher_state_dict = teacher.state_dict()
        teacher_state_dict = add_distr(teacher_state_dict)
        save_dict = {
            'teacher': teacher_state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1, 'args': args}
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(teacher, loss_obj, data_loader,
                    optimizer, lr_schedule, wd_schedule, epoch, args):
    teacher.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.finetuning_epochs)
    device = torch.device(f'cuda:{args.device}')

    for it, (batch, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i % 2 == 0:  # param_group[even_number] is regularized, else non-regularized (e.g., bias/Norm parameters)
                param_group["weight_decay"] = wd_schedule[it]

        clean_crops, corrupted_crops, *_ = batch
        
        clean_crops = [im.to(device) for im in clean_crops]
        corrupted_crops = [im.to(device) for im in corrupted_crops]
        targets = targets.to(device)

        inp = torch.cat([*corrupted_crops[0:], *clean_crops[0:]], dim=0)
        teacher_out = teacher(inp, classify=True)
        cross_ent_loss = loss_obj(teacher_out, targets.repeat(int(inp.shape[0]/args.batch_size)))
        loss = cross_ent_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), flush=True)
            sys.exit(1)
        # student update
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad:
            _ = utils.clip_gradients(teacher, args.clip_grad)
        optimizer.step()
        
        # logging
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT_teacher', parents=[get_args_parser("finetune")])
    args = parser.parse_args()
    args.teacher_dir = os.path.join(args.base_dir, args.teacher_dir_partial)
    args.teacher_dir = args.teacher_dir + "_" + args.poison_method
    args.output_dir = args.teacher_dir
    print("new args.output_dir == %s" % args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_teacher_SiT(args)
