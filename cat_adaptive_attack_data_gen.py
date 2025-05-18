import math, os, time, datetime, pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms

from datasets import datasets_utils
from datasets.poisoned_dataset import get_poisoned_dataset

from cat_adaptive_attack_model_gen import vit_poision_module, vit_no_token_prep
import utils

from train_utils import *

inv_norm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

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
    
    # pgd parameters
    parser.add_argument('--pgd_eps', type=float, default=16/255, help="pgd eps")
    parser.add_argument('--pgd_iters', type=float, default=10, help="pgd iterations")
    parser.add_argument('--cat_gamma', type=float, default=0.6, help="gamma in Towards Reliable Backdoor Attacks on Vision Transformers")
    
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


# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(poisoned_module, model_backdoor, model_target, device, data, 
                      lbl_backdoor, lbl_target, gamma):
    dat = data.clone().detach()
    dat.requires_grad = True
    features = poisoned_module(dat)
    out_backdoor = model_backdoor(features)
    out_target = model_target(features)
    loss = (1-gamma)*F.cross_entropy(out_backdoor.to(device),lbl_backdoor.to(device)) - \
        gamma*F.cross_entropy(out_target.to(device),lbl_target.to(device))
    model_backdoor.zero_grad()
    model_target.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(poisoned_module, model_backdoor, model_target, device, dat, lbl_backdoor, lbl_target, eps, alpha, 
               gamma, iters, mask, **kwargs):
    assert len(dat.shape) == 4
    def project(param_data, backup, epsilon):
        r = param_data - backup
        r = epsilon * r
        return backup + r

    x_nat = dat.clone().detach()
    x_nat_perturbed = torch.clone(x_nat)

    # Iterate over iters
    for _ in range(int(iters)):
        grad_wrt_data = gradient_wrt_data(poisoned_module, model_backdoor, model_target, device, 
                                          x_nat_perturbed, lbl_backdoor, lbl_target, gamma)
        x_nat_perturbed += torch.sign(grad_wrt_data) * alpha * mask
        perturbation_norm = torch.norm(inv_norm(x_nat_perturbed) - inv_norm(x_nat), p=float(2), dim=(1, 2, 3), keepdim=False)
        for idxx, n in enumerate(perturbation_norm):
            if float(n) > eps:
                c = eps / n
                x_nat_perturbed[idxx] = project(x_nat_perturbed[idxx], x_nat[idxx], c)
        for i in range(dat.shape[0]):
            x_nat_perturbed[i] = torch.clamp(x_nat_perturbed[i], 
                                             min=torch.amin(x_nat_perturbed[i]), 
                                             max=torch.amax(x_nat_perturbed[i]))
    return x_nat_perturbed


def gen_data(args):
    with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    device = torch.device(f'cuda:{args.device}')
    
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    cudnn.benchmark = False
    cudnn.deterministic = True
    # prepare dataset for finetuning
    transform = datasets_utils.DataAugmentationSiT(args, add_ToPILImage=False if args.poison_method != "ISSBA" else True)
    # to_small = transforms.Resize(32 if args.data_set == "CIFAR10" else 64)
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
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=False, drop_last=True, 
                collate_fn=collate_batch(args.drop_replace, args.drop_align, apply_poison=True))

    Path.mkdir(os.path.join(args.output_dir, "train_img"), exist_ok=False)
    
    poisoned_module = vit_poision_module(num_classes).to(device)

    state_dict_poisoned_module = torch.load(args.poisoned_module_chkp_pth, map_location=device)['student']
    state_dict_poisoned_module = remove_distr(state_dict_poisoned_module)
    state_dict_poisoned_module = {k[14:]: v for k, v in state_dict_poisoned_module.items() if "poison" in k}
    print(f"poisoned_module.load_state_dict, from path = {args.poisoned_module_chkp_pth}")
    msg_poison = poisoned_module.load_state_dict(state_dict_poisoned_module, strict=True)
    print(msg_poison)
    
    backdoor_discriminator = vit_no_token_prep(2, depth=4).to(device)
    target_cls = vit_no_token_prep(num_classes, depth=4).to(device)
    
    state_dict_discr = torch.load(f"checkpoints/adaptive_attack/{args.data_set}/ensemble_{args.poison_method}/checkpoint_discriminators.pth", 
                                  map_location=device)
    msg_backdoor = backdoor_discriminator.load_state_dict(remove_distr(state_dict_discr['backdoor_discriminator']), strict=True)
    msg_target = target_cls.load_state_dict(remove_distr(state_dict_discr['target_cls']), strict=True)
    print(msg_backdoor)
    print(msg_target)
    with (Path(args.output_dir) / "load_models_msg.txt").open("w") as f:
        f.write(f"-----------\npoisoned_module.load_state_dict output: \n{msg_poison}\n-----------\n")
        f.write(f"-----------\nbackdoor_discriminator.load_state_dict output: \n{msg_backdoor}\n-----------\n")
        f.write(f"-----------\ntarget_cls.load_state_dict output: \n{msg_target}\n-----------\n")
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(0, 1)
    counter = 0
    num_poisoned = 0
    for it, (batch, labels) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        *_, no_crop_imgs = batch
        no_crop_imgs = [im.to(device) for im in no_crop_imgs][0]
        backdoor_labels = [x in dataset.poisoned_set for x in range(counter, counter + len(labels))]
        backdoor_labels = torch.tensor(backdoor_labels).type(torch.LongTensor).to(device)
        for inn, x in enumerate(backdoor_labels):
            if x:
                if int(labels[inn]) != args.trigger_label and args.poison_method != "ISSBA": 
                    print(f"datapoint {counter}")
                    time.sleep(5)
                    exit(0)
                if int(inn) > len(dataset) * args.poisoning_rate + 1 and args.poison_method == "ISSBA":
                    print(f"datapoint {counter}")
                    time.sleep(5)
                    exit(0)
        target_labels = labels.to(device)
        num_poisoned += int(torch.sum(backdoor_labels).detach().cpu())
        x_adv = PGD_attack(poisoned_module, backdoor_discriminator, target_cls, device, 
                   no_crop_imgs, backdoor_labels, target_labels, eps=args.pgd_eps, 
                   alpha=1.85*args.pgd_eps/args.pgd_iters, 
                   gamma=args.cat_gamma, iters=args.pgd_iters, mask=backdoor_labels.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        for i in range(len(backdoor_labels)):
            if backdoor_labels[i]  == 1:
                with open(os.path.join(args.output_dir, f"train_img/{counter}.pickle"), "wb") as fptr:
                    pickle.dump({"img": x_adv[i].detach().cpu(),
                                 "l2": float(torch.norm(inv_norm(x_adv[i]) - inv_norm(no_crop_imgs[i]), 2)),
                                "backdoored": bool(backdoor_labels[i] == 1), 
                                "lbl": int(target_labels[i])}, fptr)
            counter += 1
        backdoor_cross_ent_loss = torch.tensor(0)
        target_cross_ent_loss = torch.tensor(0)
        if torch.sum(backdoor_labels == 1) > 0:
            with torch.no_grad():
                poisoned_features = poisoned_module(x_adv[backdoor_labels == 1])
                
                backdoor_logits = backdoor_discriminator(poisoned_features.detach().clone())
                target_logits = target_cls(poisoned_features.detach().clone())
                
                backdoor_cross_ent_loss = F.cross_entropy(backdoor_logits, backdoor_labels[backdoor_labels == 1])
                target_cross_ent_loss = F.cross_entropy(target_logits, target_labels[backdoor_labels == 1])
        
        metric_logger.update(backdoor_cross_ent_loss=backdoor_cross_ent_loss.item())
        metric_logger.update(target_cross_ent_loss=target_cross_ent_loss.item())
        metric_logger.update(overall_loss=(1-args.cat_gamma)*backdoor_cross_ent_loss.item()\
                             - args.cat_gamma*target_cross_ent_loss.item())
        metric_logger.update(frac_poisoned=num_poisoned/counter)
    
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    log_stats = {**{f'train_{k}': f"{float(v):.6g}" for k, v in train_stats.items()},
                     'epoch': 0}
    if utils.is_main_process():
        with (Path(args.output_dir) / "log_data_gen.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT_ensemble', parents=[get_args_parser_local()])
    args = parser.parse_args()
    assert args.data_set != "gtsrb"

    args.output_dir = os.path.join(args.base_dir, f"ensemble_{args.poison_method}/data")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    args.poisoned_module_chkp_pth = f"checkpoints/ensemble_all_defended/{args.data_set}/ensemble_{args.poison_method}/checkpoint.pth"
    gen_data(args)