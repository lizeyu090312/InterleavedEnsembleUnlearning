"""
This code is for supervised training the ensemble student model. 
"""
import typing, argparse, os, sys, datetime, time, math, json, torch, pickle, re
from pathlib import Path
import natsort

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from datasets import datasets_utils
from datasets.poisoned_dataset import get_poisoned_dataset, PoisonedData, DatasetFolder

from different_arch.full_pipeline import FullPipeline_Ensemble

import utils

from train_utils import *


PRESENT_KEYS = {"clean_crops", "corrupted_crops", "no_crop_imgs", "labels"}
poison_loss_list = []


class CAT_AttackDataset(Dataset):
    def __init__(self, root, transform, transform_orig, train, trigger_label, num_classes, data_set):
        super().__init__()
        assert train == True
        assert num_classes > trigger_label
        self.transform = transform
        self.root = root  # should be self.root/x.pickle
        self.trigger_label = int(trigger_label)
        paths = natsort.natsorted(os.listdir(self.root))
        self.data= dict()
        for i in range(len(paths)):
            match = re.search(r"(\d+)\.pickle", paths[i])
            if match:
                x = int(match.group(1))
            with open(os.path.join(self.root, paths[i]), "rb") as fptr:
                pkl_file = pickle.load(fptr)
                img = pkl_file["img"]
            self.data[x] = (img)
        if data_set == "CIFAR10":
            self.ds = CIFAR10(root="./datasets/cifar10", train=train,
                                  transform=transform_orig, target_transform=None)
        elif data_set == "tiny_imagenet_200":
            self.ds = DatasetFolder("./datasets/tiny_imagenet_200/train", transform=transform_orig, 
                                       extensions="jpeg", loader=utils.loader)
        
    
    def __getitem__(self, index: int):
        if index in self.data.keys():
            img = self.data[index]
            target = self.trigger_label
            if self.transform is not None:
                img = self.transform(img)
        else:
            img, target = self.ds[index]
        return img, target
    
    def __len__(self):
        return len(self.ds)


def train_SiT_supervised(args):

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
        dataset = CAT_AttackDataset(root=f"{args.output_dir}/data/train_img", 
                                    transform=transform, transform_orig=transform_orig, train=True, 
                                    trigger_label=args.trigger_label, num_classes=num_classes, 
                                    data_set=args.data_set)
    else:
        raise NotImplementedError("only supports CIFAR10 for poisoning for now")
    print(f"Successfully loaded in poisoned dataset, {len(dataset)} images")
    student, _ = return_model(args, student_or_teacher="student", num_classes=num_classes)
    data_loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, 
        collate_fn=collate_batch(args.drop_replace, args.drop_align, apply_poison=True))

    student = student.to(device)
    
    # preparing optimizer 
    optimizer_prefinetuning = torch.optim.AdamW(utils.get_params_groups(student, substr_to_exclude={"robust_module"}))  # prefinetuning
    optimizer_train = torch.optim.AdamW(utils.get_params_groups(student, substr_to_exclude={"poison_module"}))  # finetuning
    optimizer_poison = torch.optim.AdamW(utils.get_params_groups(student, substr_to_exclude={"poison_module"}))  # poisoned data

    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # init schedulers 
    lr_schedule_prefinetuning = utils.cosine_scheduler(
        base_value=args.prefinetuning_lr * (args.batch_size * utils.get_world_size()) / 256., 
        final_value=args.prefinetuning_lr * (args.batch_size * utils.get_world_size()) / 256., 
        epochs=args.prefinetuning_epochs, niter_per_ep=len(data_loader), 
        warmup_epochs=0 if args.data_set == "gtsrb" else 1)
    lr_schedule_train = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256., 
        args.min_lr, args.finetuning_epochs, len(data_loader), warmup_epochs=args.warmup_epochs)

    wd_schedule = utils.cosine_scheduler(args.weight_decay,
        args.weight_decay_end, args.finetuning_epochs, len(data_loader))

    # load in student's robust_module
    state_dict_student = torch.load("checkpoints/SiT_Small_ImageNet_ViT_student.pth",
                                    map_location=device)
    state_dict_student = remove_distr(state_dict_student)
    print(f"student.robust_module.load_state_dict, from path = checkpoints/SiT_Small_ImageNet_ViT_student.pth")
    msg_robust = student.robust_module.load_state_dict(state_dict_student, strict=False)
    print(msg_robust)
    print(f"student.poison_module.load_state_dict, from path = checkpoints/SiT_Small_ImageNet_ViT_student.pth")
    msg_poison = student.poison_module.load_state_dict(state_dict_student, strict=False)
    print(msg_poison)
    with (Path(args.output_dir) / "load_models_msg.txt").open("w") as f:
        f.write(f"-----------\nstudent.robust_module.load_state_dict output: \n{msg_robust}\n-----------\n")
        f.write(f"-----------\nstudent.poison_module.load_state_dict output: \n{msg_poison}\n-----------\n")
    # strict is False since we're loading from pretrained checkpoint

    # test_pass
    with torch.no_grad():
        print("Passing in test_inp...")
        test_inp = torch.randn((2, args.batch_size, 3, args.img_size, args.img_size)).to(device)
        _ = student(test_inp[0], confidence_threshold=args.confidence_thresh, x_no_crop=test_inp[1], mode="finetuning")
        print("Successfully passed with test_inp")

    to_restore = {"epoch": int(-1 * args.prefinetuning_epochs)}
    if args.restart_from_checkpoint == True:
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint{args.restart_epoch:04}.pth" if args.restart_epoch is not None else "checkpoint.pth")
        utils.restart_from_checkpoint(checkpoint_path, run_variables=to_restore, fp16_scaler=fp16_scaler)
        
        if to_restore['epoch'] < 0:  # next epoch to start with is in prefinetuning (i.e., prefinetuning optimizer was last active AND will continue with prefinetuning)
            optimizer_prefinetuning.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["optimizer"])
        elif to_restore['epoch'] > 0:  # next epoch to start with is finetuning AND has already trained for at least one epoch (i.e., training optimizer isnt brand new)
            optimizer_train.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["optimizer"])
        else:
            pass  # no need to load in any optimizer, since prefinetuning just finished and finetuning hasn't started
        print(f"to_restore['epoch'] == {to_restore['epoch']}")
        
        state_dict_student = torch.load(checkpoint_path, map_location=device)["student"]
        state_dict_student = remove_distr(state_dict_student)
        print(f"Loading student from {checkpoint_path} since args.restart_from_checkpoint == True...")
        print(student.load_state_dict(state_dict_student, strict=True))
    start_time = time.time()
    
    print(f"(pre)finetuning ... starting from epoch {to_restore['epoch']}")
    for epoch in range(to_restore['epoch'], args.finetuning_epochs):
        # Training
        arr = []  # preallocate all memory so that I can't CUDA OOM right after prefinetuning (denial of service, i guess?)
        if epoch < 0:
            for i in range(10):
                arr.append(torch.zeros((420, 1000, 1000)).to(device))
            optimizer, lr_schedule, mode = optimizer_prefinetuning, lr_schedule_prefinetuning, "prefinetuning"
        else:
            optimizer, lr_schedule, mode = optimizer_train, lr_schedule_train, "finetuning"
        train_stats = train_one_epoch(student, cross_ent_loss_obj, data_loader, optimizer, None if epoch < 0 else optimizer_poison, 
                                      lr_schedule, wd_schedule, epoch, fp16_scaler, args, 
                                      PoisonedData(batch_size=args.batch_size, device=device, present_keys=PRESENT_KEYS), 
                                      mode=mode, total_epochs=args.finetuning_epochs, num_classes=num_classes)
        # logs
        student_state_dict = student.state_dict()
        student_state_dict = add_distr(student_state_dict)
        save_dict = {'student': student_state_dict, 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'args': args}
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        # if args.saveckp_freq and ((epoch % args.saveckp_freq == 0 and mode == "finetuning") or \
        #                           (epoch % args.prefinetuning_saveckp_freq == 0 and mode == "prefinetuning")):
        #     utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': f"{float(v):.6g}" for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log_train.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        print(f"Epoch (prefinetuning or finetuning) {epoch} training time so far {time.time() - start_time}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student:FullPipeline_Ensemble, cross_ent_loss_obj, data_loader,
                    optimizer, optimizer_poison, lr_schedule, wd_schedule, epoch, fp16_scaler, args, 
                    poisoned_dataset:PoisonedData, mode:str, total_epochs, num_classes):
    student.train()
    if mode != "prefinetuning": 
        student.poison_module.eval()
    assert mode in {"finetuning", "prefinetuning"}
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, total_epochs)
    device = torch.device(f'cuda:{args.device}')

    didnt_update_count = 0  # counts the number of non-updates for poisoned_data
    cumulative_updated_count = 0
    for it, (batch, labels) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = 0. if mode == "prefinetuning" else wd_schedule[it]
        if optimizer_poison is not None:
            for i, param_group in enumerate(optimizer_poison.param_groups):
                scale_factor = 1.0 if len(poison_loss_list) == 0 else max(float(6 - math.exp(-(math.log(num_classes) - poison_loss_list[-1]/math.sqrt(2)))), 0.2)
                param_group["lr"] = lr_schedule[it] * scale_factor
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[it]
        clean_crops, corrupted_crops, _, no_crop_imgs = batch
        
        clean_crops = [im.to(device) for im in clean_crops]
        corrupted_crops = [im.to(device) for im in corrupted_crops]
        no_crop_imgs = [im.to(device) for im in no_crop_imgs]
        labels = labels.to(device)
        
        def _student_forward(inp, inp_no_crop, 
                             mode:str, get_poisoned_inp:bool):
            out_ = student(inp, confidence_threshold=args.confidence_thresh, x_no_crop=inp_no_crop, 
                           mode=mode)
            poison_out, robust_out, weight = out_["poison_out"], out_["robust_out"], out_["weight"]
            temp_d = {True: ":weight_corru:", False: ":weight_clean:"}
            if mode == "finetuning":
                assert poison_out is not None and robust_out is not None and weight is not None
                poison_logits_noise_variance = 0. if hasattr(args, "poison_logits_noise_variance") == False else \
                    args.poison_logits_noise_variance
                poison_out_softmax = F.softmax(poison_out+torch.randn_like(poison_out)*poison_logits_noise_variance, dim=-1)
                poison_out_softmax_max = torch.amax(torch.softmax(poison_out, dim=-1), dim=-1)
                poison_out_softmax_filtered = torch.where((poison_out_softmax_max > args.confidence_thresh).unsqueeze(1), 
                                                          input=poison_out_softmax, other=float(1/poison_out.shape[-1]))
                poisoned_inp = {k: torch.zeros(0) for k in PRESENT_KEYS}
                if get_poisoned_inp == True and mode == "finetuning":
                    poison_idx = (poison_out_softmax_max > args.confidence_thresh).flatten()  # poisoned_samples
                    poisoned_inp['clean_crops'], poisoned_inp['corrupted_crops'], poisoned_inp['no_crop_imgs'], poisoned_inp['labels'] = \
                        clean_crops[0][poison_idx], corrupted_crops[0][poison_idx], no_crop_imgs[0][poison_idx], labels[poison_idx]
                
                # set poison_out_softmax_filtered[i] to 1/poison_out.shape[-1] if diff is smaller than threshold. 
                return {"logits": torch.log(poison_out_softmax_filtered) + torch.log(F.softmax(robust_out, dim=-1)) * weight, 
                        "poisoned_inp": poisoned_inp}
            elif mode == "prefinetuning":  # not including robust_out, effectively freezing the robust module
                return {"logits": poison_out, "poisoned_inp": None}
            else: raise NotImplementedError

        student_in_cor, student_in_no_crop = torch.cat(corrupted_crops[0:]), torch.cat(no_crop_imgs[0:])
        # poisoned_data is only dependent on poison_out, which is in turn only dependent on student_in_no_crop -> only need it once
        _ret = _student_forward(student_in_cor, student_in_no_crop, 
                                mode, get_poisoned_inp=True)
        out_cor, poisoned_inp = _ret["logits"], _ret["poisoned_inp"]
        
        student_in_clean = torch.cat(clean_crops[0:])
        _ret = _student_forward(student_in_clean, student_in_no_crop,
                                mode, get_poisoned_inp=False)
        out_clean = _ret["logits"]

        all_logits = torch.cat((out_clean, out_cor), dim=0)
        cross_ent_loss = cross_ent_loss_obj(all_logits, labels.repeat(int(all_logits.shape[0]/args.batch_size)))
        if mode != "prefinetuning":
            loss = cross_ent_loss
        else:  # Local Gradient Ascent (see ABL paper, backdoor_isolation.py)
            # loss = torch.sign(cross_ent_loss - args.gamma) * cross_ent_loss
            if args.gradient_ascent_type == 'LGA':
                # add Local Gradient Ascent(LGA) loss
                loss = torch.sign(cross_ent_loss - args.gamma_or_flooding) * cross_ent_loss

            elif args.gradient_ascent_type == 'Flooding':
                # add flooding loss
                loss = (cross_ent_loss - args.gamma_or_flooding).abs() + args.gamma_or_flooding
            
            elif args.gradient_ascent_type == 'no-method':
                loss = cross_ent_loss
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), flush=True)
            sys.exit(1)

        # student update
        def student_update(args, student, optimizer, fp16_scaler, loss):
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                if args.clip_grad:
                    _ = utils.clip_gradients(student, args.clip_grad)
                
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
        
        student_update(args, student, optimizer, fp16_scaler, loss)
        cross_ent_loss_poisoned_data = torch.tensor(0)
        if mode == "finetuning":
            didnt_update_count += 1
            if len(poisoned_inp["corrupted_crops"]) > 0:
                clean_crops_poison, corrupted_crops_poison, no_crop_imgs_poison, labels_poison = \
                    poisoned_dataset.add_sample(poisoned_inp)
                if torch.is_tensor(labels_poison) == True:
                    all_three_poison = torch.concatenate((clean_crops_poison, corrupted_crops_poison, no_crop_imgs_poison), dim=0)
                    poison_data_ret = student(all_three_poison, confidence_threshold=args.confidence_thresh, 
                                              x_no_crop=no_crop_imgs_poison, mode=mode)
                    # note: confidence_thresh_logits does not do anything since poison_data_ret["weights"] is not used 
                    poison_data_ret_robust_out = poison_data_ret["robust_out"]  # robust_model's output logits
                    print(f"poison_data_ret_robust_out.shape[0]={poison_data_ret_robust_out.shape[0]}, batch_size={args.batch_size}, optimizer_poison lr={float(optimizer_poison.param_groups[0]["lr"]):.4g}, orig_lr={float(optimizer.param_groups[0]["lr"]):.4g}")
                    cross_ent_loss_poisoned_data = -1 * cross_ent_loss_obj(poison_data_ret_robust_out, 
                                                                           labels_poison.type(torch.long).repeat(int(poison_data_ret_robust_out.shape[0]/args.batch_size)).to(device))
                    poison_loss_list.append(float(torch.abs(cross_ent_loss_poisoned_data)))
                    # gradient ascent
                    student_update(args, student, optimizer_poison, fp16_scaler, cross_ent_loss_poisoned_data)
                    cross_ent_loss_poisoned_data *= didnt_update_count
                    didnt_update_count = 0
                    cumulative_updated_count += 1
            
        # logging
        metric_logger.update(cross_ent_loss=cross_ent_loss.item())
        metric_logger.update(cross_ent_loss_poisoned_data=cross_ent_loss_poisoned_data.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if optimizer_poison is not None:
            metric_logger.update(lr=optimizer_poison.param_groups[0]["lr"])
            # ^^ this is a mistake (which will remain in the program). This should be named differently, 
            # e.g. lr_poison= etc this mistake explains why train_lr is different across different runs. 
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(cumulative_updated_count=cumulative_updated_count)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT_ensemble', parents=[get_args_parser("finetune")])
    args = parser.parse_args()
    args.teacher_dir = os.path.join(args.base_dir, args.teacher_dir_partial)
    args.teacher_dir = args.teacher_dir + "_" + args.poison_method
    args.output_dir = os.path.join(args.base_dir, args.output_dir_partial)
    args.output_dir = args.output_dir + "_" + args.poison_method
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_SiT_supervised(args)
