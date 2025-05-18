import argparse
import json
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from datasets import datasets_utils

from different_arch import full_pipeline 
from different_arch import vit_arch as vits
from different_arch.info import MODEL_NAME_TO_FN_DICT


import utils


def get_args_parser(mode):

    parser = argparse.ArgumentParser('SiT', add_help=False)

    # The BooleanOptionalAction will generate 2 options: --debug --> True; --no-debug --> False
    parser.add_argument('--device', type=int, default=1, help="GPU device (not distributed).")


    # parser.add_argument('--dev', type=bool, default=False, help='Developing features', action=argparse.BooleanOptionalAction)
    parser.add_argument("--gradient_ascent_type", type=str, default="no-method", 
                             choices=["LGA", "Flooding", "no-method"],
                             help="Amount of data used for unlearning (see ABL)")
    parser.add_argument("--gamma_or_flooding", type=float, default=0, 
                             help="Amount of data used for unlearning (see ABL)")

    # Data augmentation and reconstruction parameters
    parser.add_argument('--drop_perc', type=float, default=0.3, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0.0, help='Drop X percentage of the input image')
    
    parser.add_argument('--drop_align', type=int, default=1, help='Align drop with patches; Set to patch size to align corruption with patches')
    parser.add_argument('--drop_type', type=str, default='zeros-noise', help='Drop Type.')

    parser.add_argument('--no_color_jitter', type=bool, default=False, help='Apply color jitter?', action=argparse.BooleanOptionalAction)
    
    # Model parameters
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Poisoning parameters
    # parser.add_argument('--apply_poison', type=bool, default=False, help='Whether or not to use the BadNet poisoned dataset.', action=argparse.BooleanOptionalAction)
    parser.add_argument('--poison_method', type=str, choices=["BadNets-white", "BadNets-pattern", "ISSBA", "BATT", "Blended", "Trojan-WM", "Trojan-SQ", "Smooth", "l0-inv", "l2-inv", "SIG", "no-attack", "WaNet"], help="Type of trigger")
    parser.add_argument('--trigger_label', type=int, default=1, help="Poisoned data's target is changed to trigger_label")
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help="Fraction of poisoned data (relative to the entire dataset, e.g. 50k images for CIFAR10)")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.1)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument("--prefinetuning_lr", default=2e-4, type=float, help="Learning rate during prefinetuning.")
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target LR at the end of finetuning.")

    # parser.add_argument("--gamma", default=0.5, type=float, help="Gamma in LGA (local gradient ascent), in ABL paper.")
    parser.add_argument("--confidence_thresh", default=0.95, type=float, help="Confidence threshold for poison_model, possible values in (0, 1). Currently used in _student_forward and FullPipeline_Ensemble's forward.")
    parser.add_argument("--prefinetuning_epochs", default=5, type=int, help="Number of epochs before warmup (prefinetuning stuff, potentially 0 epochs of warmup)")
    parser.add_argument("--warmup_epochs", default=1, type=int, help="Number of epochs for the linear learning-rate warm up for finetuning.")
    parser.add_argument('--finetuning_epochs', default=10, type=int, help='Number of epochs of finetuning after prefinetuning_epochs.')

    parser.add_argument('--restart_epoch', type=int, default=None, help='From which epoch to restart finetuning? Used if restart_from_checkpoint==True')
    parser.add_argument('--restart_from_checkpoint', type=bool, default=False, help='Restart finetuning?', action=argparse.BooleanOptionalAction)

    # Dataset
    parser.add_argument('--data_set', default='CIFAR10', type=str, 
                        choices=['CIFAR10', 'gtsrb', 'tiny_imagenet_200'], help='Name of the dataset.')
    # parser.add_argument('--data_location', default='./datasets', type=str, help='Dataset location.')
    parser.add_argument('--img_size', type=int, default=224, help="size of input img (CIFAR10->32, ImageNet->224)")
    parser.add_argument('--patch_size', type=int, default=16, help="size of 1x patch (CIFAR10->4, ImageNet->16)")

    parser.add_argument('--base_dir', type=str, required=True, help="base directory for all checkpoints")  # e.g. "checkpoints/ensemble_from_checkpoint_gtsrb"
    parser.add_argument('--output_dir_partial', default="ensemble", type=str, help='Partial path to save logs and checkpoints.')
    parser.add_argument('--teacher_dir_partial', type=str, default="teacher", help="Partial path to teacher dir")
    parser.add_argument('--saveckp_freq', default=2, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--prefinetuning_saveckp_freq', default=2, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--distributed', type=bool, default=False, help='Distributed finetuning?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_workers', default=3, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="set up distributed finetuning")
    parser.add_argument("--local_rank", default=0, type=int)
    return parser


# replace from other images
class collate_batch(object): 
    def __init__(self, drop_replace=0., drop_align=1, apply_poison=False):
        self.drop_replace = drop_replace
        self.drop_align = drop_align
        self.apply_poison = apply_poison
        
    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        # batch[0], batch[1] = (clean_crops, corrupted_crops, masked_crops), img_label
        # batch[0][1][k] is the k-th corrupted image (out of 2 images in total) in the corrupted crops list
        # batch[0][2][k] is the k-th mask (out of 2 masks in total) in the masked_crops list
        # batch[0][0][k] is the k-th clean image (out of 2 imgs in total) in the clean_crops list
        if self.drop_replace > 0 and (self.apply_poison == False):
            batch[0][1][0], batch[0][2][0] = datasets_utils.GMML_replace_list(batch[0][0][0], batch[0][1][0], batch[0][2][0],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
            batch[0][1][1], batch[0][2][1] = datasets_utils.GMML_replace_list(batch[0][0][1], batch[0][1][1], batch[0][2][1],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
        
        return batch


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def remove_distr(old_distr_state_dict:dict):
    for k, _ in old_distr_state_dict.items():
        if k[0:7] != "module.": 
            return old_distr_state_dict
    new_state_dict = OrderedDict()
    for k, v in old_distr_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def add_distr(old_non_distr_state_dict:dict):
    new_state_dict = {}
    for k, v in old_non_distr_state_dict.items():
        assert k[0:7] != "module."
        name = "module." + k  # add `module.` to the key
        new_state_dict[name] = v
    return new_state_dict


def return_model(args, student_or_teacher:str, num_classes, **kwargs) \
        -> tuple[full_pipeline.FullPipeline_Ensemble|vits.VisionTransformer, dict]:
    assert student_or_teacher in {"student", "teacher"}, "check student_or_teacher parameter"
    poison_depth = 1 if hasattr(args, "poison_module_depth") == False else args.poison_module_depth
    model_arch = "vit" if hasattr(args, "model_arch") == False else args.model_arch
    
    robust_module = MODEL_NAME_TO_FN_DICT[model_arch](
        num_classes=num_classes, patch_size=args.patch_size, 
        depth=12 if model_arch != "cait" else 24, img_size=args.img_size, drop_path_rate=args.drop_path_rate)

    # poison_module always ViT architecture
    poison_module = vits.vit_small_patch16(
        num_classes=num_classes, patch_size=args.patch_size, 
        depth=poison_depth, img_size=args.img_size, drop_path_rate=0)

    if student_or_teacher == "teacher":
        ret_model = robust_module
    elif student_or_teacher == "student":
        ret_model = full_pipeline.FullPipeline_Ensemble(
            robust_module=robust_module, poison_module=poison_module, **kwargs)
    return ret_model, kwargs


def load_args(arg_file_path):
    with open(arg_file_path, "r") as fptr:
        data_dict = json.load(fptr)
    return type('testclass', (object,), data_dict)()
