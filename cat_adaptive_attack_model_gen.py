import math, os, time, datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from different_arch.vit_arch import VisionTransformer, vit_small_patch16

from datasets import datasets_utils
from datasets.poisoned_dataset import get_poisoned_dataset, PoisonedData

from different_arch.full_pipeline import FullPipeline_Ensemble

import utils

from train_utils import *

def get_args_parser_local():

    parser = argparse.ArgumentParser('SiT', add_help=False)

    # The BooleanOptionalAction will generate 2 options: --debug --> True; --no-debug --> False
    parser.add_argument('--device', type=int, default=1, help="GPU device")

    # Data augmentation and reconstruction parameters
    parser.add_argument('--drop_perc', type=float, default=0.3, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0.0, help='Drop X percentage of the input image')
    parser.add_argument('--drop_align', type=int, default=1, help='Align drop with patches; Set to patch size to align corruption with patches')
    parser.add_argument('--drop_type', type=str, default='zeros-noise', help='Drop Type.')
    parser.add_argument('--no_color_jitter', type=bool, default=False, help='Apply color jitter?', action=argparse.BooleanOptionalAction)
    
    # Poisoning parameters
    parser.add_argument('--poison_method', type=str, choices=["BadNets-white", "BadNets-pattern", "ISSBA", "BATT", "Blended", "Trojan-WM", "Trojan-SQ", "Smooth", "l0-inv", "l2-inv", "SIG", "no-attack", "WaNet"], help="Type of trigger")
    parser.add_argument('--trigger_label', type=int, default=1, help="Poisoned data's target is changed to trigger_label")
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help="Fraction of poisoned data (relative to the entire dataset, e.g. 50k images for CIFAR10)")
    parser.add_argument('--img_size', type=int, default=224, help="size of input img (CIFAR10->32, ImageNet->224)")
    
    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target LR at the end of finetuning.")

    parser.add_argument("--warmup_epochs", default=1, type=int, help="Number of epochs for the linear learning-rate warm up for finetuning.")
    parser.add_argument('--finetuning_epochs', default=10, type=int, help='Number of epochs of finetuning.')

    # Dataset
    parser.add_argument('--data_set', default='CIFAR10', type=str, 
                        choices=['CIFAR10', 'gtsrb', 'tiny_imagenet_200'], help='Name of the dataset.')
    parser.add_argument('--base_dir', type=str, required=True, help="base directory for all checkpoints")  # e.g. "checkpoints/ensemble_from_checkpoint_gtsrb"
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=3, type=int, help='Number of data loading workers per GPU.')
    return parser


class VisionTransformer_OnlyAttn1stLayer(VisionTransformer):
    def forward(self, x, **kwargs):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
            return x


class VisionTransformer_NoTokenPrep(VisionTransformer):
    def forward(self, x, **kwargs):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head( torch.cat( (x[:, 0], torch.mean(x[:, 1:], dim=1)), dim=1 ) )   

        
def vit_poision_module(num_classes, **kwargs) -> VisionTransformer_OnlyAttn1stLayer:
    model = VisionTransformer_OnlyAttn1stLayer(
        img_size=[224], patch_size=16, embed_dim=384, depth=1, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, 
        num_classes=num_classes, **kwargs)
    return model


def vit_no_token_prep(num_classes, depth, **kwargs) -> VisionTransformer_NoTokenPrep:
    model = VisionTransformer_NoTokenPrep(
        img_size=[224], patch_size=16, embed_dim=384, depth=depth, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, 
        num_classes=num_classes, **kwargs)
    return model


def train_discriminators(args):
    with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    device = torch.device(f'cuda:{args.device}')

    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    cudnn.benchmark = False
    cudnn.deterministic = True
    # prepare dataset for finetuning
    cross_ent_loss_obj = nn.CrossEntropyLoss()
    transform = datasets_utils.DataAugmentationSiT(args, add_ToPILImage=False if args.poison_method != "ISSBA" else True)

    if args.data_set == 'CIFAR10' or args.data_set == 'gtsrb' or args.data_set == 'tiny_imagenet_200':
        dataset = get_poisoned_dataset(method=args.poison_method, dataset_name=args.data_set, train=True, transform=transform, 
                                        trigger_label=args.trigger_label, poisoning_rate=args.poisoning_rate)
        match args.data_set:
            case "CIFAR10":
                num_classes = 10
            case "gtsrb":
                num_classes = 43
            case "tiny_imagenet_200":
                num_classes = 200
    else:
        raise NotImplementedError("only supports CIFAR10 for poisoning for now")
    print("Successfully loaded in poisoned dataset")
    data_loader = torch.utils.data.DataLoader(dataset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True, 
                collate_fn=collate_batch(args.drop_replace, args.drop_align, apply_poison=True))

    poisoned_module = vit_poision_module(num_classes).to(device)

    state_dict_poisoned_module = torch.load(args.checkpoint_path, map_location=device)['student']
    state_dict_poisoned_module = remove_distr(state_dict_poisoned_module)
    state_dict_poisoned_module = {k[14:]: v for k, v in state_dict_poisoned_module.items() if "poison" in k}
    print(f"poisoned_module.load_state_dict, from path = {args.checkpoint_path}")
    msg_poison = poisoned_module.load_state_dict(state_dict_poisoned_module, strict=False)
    print(msg_poison)
    
    backdoor_discriminator = vit_no_token_prep(2, depth=4).to(device)
    target_cls = vit_no_token_prep(num_classes, depth=4).to(device)
    
    state_dict_pretrained = torch.load("checkpoints/SiT_Small_ImageNet_ViT_student.pth", map_location=device)
    state_dict_pretrained = remove_distr(state_dict_pretrained)
    msg_backdoor = backdoor_discriminator.load_state_dict(state_dict_pretrained, strict=False)
    msg_target = target_cls.load_state_dict(state_dict_pretrained, strict=False)
    print(msg_backdoor)
    print(msg_target)
    
    
    with (Path(args.output_dir) / "load_models_msg.txt").open("w") as f:
        f.write(f"-----------\npoisoned_module.load_state_dict output: \n{msg_poison}\n-----------\n")
        f.write(f"-----------\nbackdoor_discriminator.load_state_dict output: \n{msg_backdoor}\n-----------\n")
        f.write(f"-----------\ntarget_cls.load_state_dict output: \n{msg_target}\n-----------\n")
    
    
    # preparing optimizer 
    optimizer_backdoor = torch.optim.AdamW(utils.get_params_groups(backdoor_discriminator, substr_to_exclude={}))  # prefinetuning
    optimizer_target = torch.optim.AdamW(utils.get_params_groups(target_cls, substr_to_exclude={}))  # finetuning

    # init schedulers 
    lr_schedule_tune = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256., 
        args.min_lr, args.finetuning_epochs, len(data_loader), warmup_epochs=args.warmup_epochs)

    wd_schedule = utils.cosine_scheduler(args.weight_decay,
        args.weight_decay, args.finetuning_epochs, len(data_loader))
    
    start_time = time.time()
    for epoch in range(0, args.finetuning_epochs):
        # Training
        train_stats = train_one_epoch(backdoor_discriminator, target_cls, 
                                      optimizer_backdoor, optimizer_target,
                                      poisoned_module,
                                      cross_ent_loss_obj, data_loader, 
                                      lr_schedule_tune, wd_schedule, epoch, args,  
                                      total_epochs=args.finetuning_epochs)
        target_cls_state_dict = target_cls.state_dict()
        backdoor_discriminator_state_dict = backdoor_discriminator.state_dict()
        target_cls_state_dict = add_distr(target_cls_state_dict)
        backdoor_discriminator_state_dict = add_distr(backdoor_discriminator_state_dict)
        
        save_dict = {'target_cls': target_cls_state_dict, 
                     'backdoor_discriminator': backdoor_discriminator_state_dict, 
                     'optimizer_backdoor': optimizer_backdoor.state_dict(), 
                     'optimizer_target': optimizer_target.state_dict(),
                     'epoch': epoch + 1, 'args': args}
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint_discriminators.pth'))
        log_stats = {**{f'train_{k}': f"{float(v):.6g}" for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        print(f"Epoch (prefinetuning or finetuning) {epoch} training time so far {time.time() - start_time}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return 


def train_one_epoch(backdoor_discriminator, target_cls, 
                                      optimizer_backdoor, optimizer_target,
                                      poisoned_module,
                                      cross_ent_loss_obj, data_loader, 
                                      lr_schedule, wd_schedule, epoch, args,  
                                      total_epochs):
    def _update(args, model, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad:
            _ = utils.clip_gradients(model, args.clip_grad)
        optimizer.step()
        return
    
    backdoor_discriminator.train()
    target_cls.train()
    poisoned_module.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, total_epochs)
    device = torch.device(f'cuda:{args.device}')

    for it, (batch, labels) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer_backdoor.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]
        for i, param_group in enumerate(optimizer_target.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]
        *_, no_crop_imgs = batch

        no_crop_imgs = [im.to(device) for im in no_crop_imgs][0]
        backdoor_labels = torch.tensor(labels == args.trigger_label).type(torch.LongTensor).to(device)
        target_labels = labels.to(device)
        
        poisoned_features = poisoned_module(no_crop_imgs)
        
        backdoor_logits = backdoor_discriminator(poisoned_features.detach().clone())
        target_logits = target_cls(poisoned_features.detach().clone())
        
        backdoor_cross_ent_loss = cross_ent_loss_obj(backdoor_logits, backdoor_labels)
        target_cross_ent_loss = cross_ent_loss_obj(target_logits, target_labels)

        _update(args, backdoor_discriminator, backdoor_cross_ent_loss, optimizer_backdoor)
        _update(args, target_cls, target_cross_ent_loss, optimizer_target)
        
        metric_logger.update(backdoor_cross_ent_loss=backdoor_cross_ent_loss.item())
        metric_logger.update(target_cross_ent_loss=target_cross_ent_loss.item())
        metric_logger.update(lr=optimizer_backdoor.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT_ensemble', parents=[get_args_parser_local()])
    args = parser.parse_args()
    assert args.data_set != "gtsrb"

    args.output_dir = os.path.join(args.base_dir, "ensemble")
    args.output_dir = args.output_dir + "_" + args.poison_method
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.checkpoint_path = f"checkpoints/ensemble_all_defended/{args.data_set}/ensemble_{args.poison_method}/checkpoint.pth"
    train_discriminators(args)
