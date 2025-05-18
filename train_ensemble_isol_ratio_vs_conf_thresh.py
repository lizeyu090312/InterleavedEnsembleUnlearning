"""
The goal is to find out how well the poisoned data is detected using isolation ratio vs confidence_thresh
given different poisoning rates. 
"""
import numpy as np
import os, tqdm, argparse, sys, datetime, time, math, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score

from different_arch import vit_arch as vit
from different_arch.full_pipeline import FullPipeline_Ensemble

from datasets import datasets_utils
from datasets.poisoned_dataset import get_poisoned_dataset

import utils

from train_utils import *


PRESENT_KEYS = {"clean_crops", "corrupted_crops", "no_crop_imgs", "labels"}
poison_loss_list = []


def get_conf_mat(args, student):
    student.eval()
    device = torch.device(f'cuda:{args.device}')
    loss_obj = nn.CrossEntropyLoss(reduction='none')

    transform = datasets_utils.basic_transforms(
        args, train=False, add_ToPILImage=False if args.poison_method != "ISSBA" else True)
    dataset = get_poisoned_dataset(method=args.poison_method, dataset_name=args.data_set, train=True, transform=transform, 
                                        trigger_label=args.trigger_label, poisoning_rate=args.poisoning_rate)
    batch_size = 256
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False,
        batch_size=batch_size, num_workers=args.num_workers, pin_memory=False, drop_last=False)

    path_to_shuffled_indices = f"./shuffle_indices/{args.data_set}_shuffle_indices_train_True.npy"
    num_imgs_poisoned = int(len(dataset)*args.poisoning_rate)
    poisoned_indices_train = frozenset(np.load(path_to_shuffled_indices)[0:num_imgs_poisoned])
    if args.poison_method == "ISSBA":
        poisoned_indices_train = frozenset(range(num_imgs_poisoned))  
    true_poisoned = np.array([1 if i in poisoned_indices_train else 0 for i in range(len(dataset))])

    losses_record = []
    conf_thresh_pred_poisoned = np.zeros_like(true_poisoned)
    isolation_ratio_pred_poisoned = np.zeros_like(true_poisoned)
    
    print("Successfully loaded in test poisoned dataset")
    for it, (batch, labels) in tqdm.tqdm(enumerate(data_loader)):        
        batch, labels = batch.to(device), labels.to(device)
        labels = labels.to(device)
        model_in = batch
        
        cor_out = student(model_in, mode="finetuning", confidence_threshold=args.confidence_thresh)

        start_idx, end_idx = it * batch_size, min((it+1) * batch_size, len(dataset))
        weight = cor_out["weight"].squeeze().cpu().numpy().astype(np.int8)  # weight is multiplied on robust_out 
        # => where weight is zero, the item is poisoned
        # "logits": torch.log(poison_out_softmax_filtered) + torch.log(F.softmax(robust_out, dim=-1)) * weight
        inv_weight = 1 - weight
        conf_thresh_pred_poisoned[start_idx:end_idx] = inv_weight

        logits_poisoned = cor_out["poison_out"]
        cross_ent_loss_poisoned = loss_obj(logits_poisoned, labels)
        losses_record.extend(cross_ent_loss_poisoned.detach().cpu().tolist())
        
    losses_idx = np.argsort(np.array(losses_record))
    perm = losses_idx[0:int(len(losses_idx) * args.isolation_ratio)]

    for k in perm:
        isolation_ratio_pred_poisoned[k] = 1

    C = confusion_matrix(
        y_true=true_poisoned, y_pred=isolation_ratio_pred_poisoned, normalize="true")
    precision = precision_score(y_true=true_poisoned, y_pred=isolation_ratio_pred_poisoned)
    acc = accuracy_score(y_true=true_poisoned, y_pred=isolation_ratio_pred_poisoned)
    f1 = f1_score(y_true=true_poisoned, y_pred=isolation_ratio_pred_poisoned)
    # TN: {C[0, 0]:.6f}, FN: {C[1, 0]:.6f}, TP: {C[1, 1]:.6f}, FP: {C[0, 1]:.6f}
    with open(os.path.join(args.output_dir, "results_log.txt"), 'a') as fptrrr:
        fptrrr.write(f"tnr,fnr,tpr,fpr,precision,accuracy,f1,isolation_ratio\n"
                     f"{C[0,0]:.6f},{C[1,0]:.6f},{C[1,1]:.6f},{C[0,1]:.6f},{precision:.6f},{acc:.6f},{f1:.6f},{args.isolation_ratio:g}\n")
    
    C = confusion_matrix(
        y_true=true_poisoned, y_pred=conf_thresh_pred_poisoned, normalize="true")
    precision = precision_score(y_true=true_poisoned, y_pred=conf_thresh_pred_poisoned)
    acc = accuracy_score(y_true=true_poisoned, y_pred=conf_thresh_pred_poisoned)
    f1 = f1_score(y_true=true_poisoned, y_pred=conf_thresh_pred_poisoned)
    with open(os.path.join(args.output_dir, "results_log.txt"), 'a') as fptrrr:
        fptrrr.write(f"tnr,fnr,tpr,fpr,precision,accuracy,f1,confidence_thresh\n"
                     f"{C[0,0]:.6f},{C[1,0]:.6f},{C[1,1]:.6f},{C[0,1]:.6f},{precision:.6f},{acc:.6f},{f1:.6f},{args.confidence_thresh:g}\n")
    return


def train_SiT_supervised(args):

    with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    device = torch.device(f'cuda:{args.device}')

    if args.distributed == True:
        utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    cudnn.benchmark = False
    cudnn.deterministic = True
    # prepare dataset for finetuning
    cross_ent_loss_obj = nn.CrossEntropyLoss()
    # simclr_loss_obj = SimCLR(device, distributed=args.distributed, temp=args.simclr_temp)
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

    student, _ = return_model(args, student_or_teacher="student", num_classes=num_classes)
    student.robust_module = vit.MyIdentity()
    
    if args.distributed == True:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(dataset,
            sampler=sampler, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, drop_last=True, 
            collate_fn=collate_batch(args.drop_replace, args.drop_align, apply_poison=True))
            # collate_batch only does cross-image collating if (self.drop_replace > 0 and self.apply_poison == False)
        print(f"Data loaded: there are {len(dataset)} images.")
        student = student.cuda()

        # synchronize batch norms
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)

        # we need DDP wrapper to have synchro batch norms working...
        student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False)
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True, 
            collate_fn=collate_batch(args.drop_replace, args.drop_align, apply_poison=True))

        student = student.to(device)
    
    # preparing optimizer 
    optimizer_prefinetuning = torch.optim.AdamW(utils.get_params_groups(student, substr_to_exclude={"robust_module"}))

    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # init schedulers 
    lr_schedule_prefinetuning = utils.cosine_scheduler(
        base_value=args.prefinetuning_lr * (args.batch_size * utils.get_world_size()) / 256., 
        final_value=args.prefinetuning_lr * (args.batch_size * utils.get_world_size()) / 256., 
        epochs=args.prefinetuning_epochs, niter_per_ep=len(data_loader), 
        warmup_epochs=0 if args.data_set == "gtsrb" else 1)

    wd_schedule = utils.cosine_scheduler(args.weight_decay,
        args.weight_decay_end, args.finetuning_epochs, len(data_loader))

    # load in student's robust_module
    state_dict_student = torch.load("checkpoints/SiT_Small_ImageNet_ViT_student.pth",
                                    map_location=device)
    if args.distributed == False:
        state_dict_student = remove_distr(state_dict_student)
    print(f"student.robust_module.load_state_dict, from path = checkpoints/SiT_Small_ImageNet_ViT_student.pth")
    msg_robust = student.robust_module.load_state_dict(state_dict_student, strict=False)
    # print(msg_robust)
    print(f"student.poison_module.load_state_dict, from path = checkpoints/SiT_Small_ImageNet_ViT_student.pth")
    msg_poison = student.poison_module.load_state_dict(state_dict_student, strict=False)
    # print(msg_poison)
    with (Path(args.output_dir) / "load_models_msg.txt").open("w") as f:
        f.write(f"-----------\nstudent.robust_module.load_state_dict output: \n{msg_robust}\n-----------\n")
        f.write(f"-----------\nstudent.poison_module.load_state_dict output: \n{msg_poison}\n-----------\n")
    # strict is False since we're loading from pretrained checkpoint
    if args.test_only == True:
        state_d = torch.load(os.path.join(args.output_dir, "checkpoint.pth"),
                                    map_location=device)["student"]
        if args.distributed == False: 
            state_d = remove_distr(state_d)
        msg_test_only = student.load_state_dict(state_d, strict=True)
        with (Path(args.output_dir) / "load_models_msg.txt").open("a") as f:
            f.write(f"-----------\nargs.test_only == True:\nstudent.load_state_dict output: \n{msg_test_only}\n-----------\n")

    # test_pass
    print("Passing in test_inp...")
    test_inp = torch.randn((2, args.batch_size, 3, args.img_size, args.img_size)).to(device)
    _ = student(test_inp[0], confidence_threshold=args.confidence_thresh, x_no_crop=test_inp[1], mode="finetuning")
    print("Successfully passed with test_inp")

    to_restore = {"epoch": int(-1 * args.prefinetuning_epochs)}
    start_time = time.time()

    if args.test_only == False:
        print(f"(pre)finetuning ... starting from epoch {to_restore['epoch']}")
        for epoch in range(to_restore['epoch'], 0):
            if args.distributed == True:
                data_loader.sampler.set_epoch(epoch)
            # Training
            optimizer, lr_schedule, mode = optimizer_prefinetuning, lr_schedule_prefinetuning, "prefinetuning"
            train_stats = train_one_epoch(student, cross_ent_loss_obj, data_loader, optimizer, 
                                        lr_schedule, wd_schedule, epoch, fp16_scaler, args, 
                                        mode=mode, total_epochs=args.finetuning_epochs)
            # logs
            student_state_dict = student.state_dict()
            if args.distributed == False:
                student_state_dict = add_distr(student_state_dict)
            save_dict = {'student': student_state_dict, 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'args': args}
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
            if args.saveckp_freq and ((epoch % args.saveckp_freq == 0 and mode == "finetuning") or \
                                    (epoch % args.prefinetuning_saveckp_freq == 0 and mode == "prefinetuning")):
                utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            log_stats = {**{f'train_{k}': f"{float(v):.6g}" for k, v in train_stats.items()},
                        'epoch': epoch}
            if utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            print(f"Epoch (prefinetuning or finetuning) {epoch} training time so far {time.time() - start_time}")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    else:
        print("Testing only since args.test_only == True")
    print("Testing...")
    get_conf_mat(args, student)
    return


def train_one_epoch(student:FullPipeline_Ensemble, cross_ent_loss_obj, data_loader,
                    optimizer, lr_schedule, wd_schedule, epoch, fp16_scaler, args, 
                    mode:str, total_epochs):
    student.train()
    if mode != "prefinetuning": 
        student.poison_module.eval()
    assert mode in {"finetuning", "prefinetuning"}
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, total_epochs)
    device = torch.device(f'cuda:{args.device}')

    for it, (batch, labels) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = 0. if mode == "prefinetuning" else wd_schedule[it]
        clean_crops, corrupted_crops, _, no_crop_imgs = batch
        if args.distributed == True:
            clean_crops = [im.cuda(non_blocking=True) for im in clean_crops]
            corrupted_crops = [im.cuda(non_blocking=True) for im in corrupted_crops]
            no_crop_imgs = [im.cuda(non_blocking=True) for im in no_crop_imgs]
            labels = labels.cuda(non_blocking=True)
        else:
            clean_crops = [im.to(device) for im in clean_crops]
            corrupted_crops = [im.to(device) for im in corrupted_crops]
            no_crop_imgs = [im.to(device) for im in no_crop_imgs]
            labels = labels.to(device)
        
        def _student_forward(inp, inp_no_crop):
            out_ = student(inp, confidence_threshold=args.confidence_thresh, x_no_crop=inp_no_crop, 
                           mode=mode)
            poison_out = out_["poison_out"]
            if mode == "prefinetuning":  # not including robust_out, effectively freezing the robust module
                return {"logits": poison_out, "poisoned_inp": None}
            else: raise NotImplementedError

        student_in_cor, student_in_no_crop = torch.cat(corrupted_crops[0:]), torch.cat(no_crop_imgs[0:])
        # poisoned_data is only dependent on poison_out, which is in turn only dependent on student_in_no_crop -> only need it once
        _ret = _student_forward(student_in_cor, student_in_no_crop)
        out_cor = _ret["logits"]
        
        student_in_clean = torch.cat(clean_crops[0:])
        _ret = _student_forward(student_in_clean, student_in_no_crop)
        out_clean = _ret["logits"]

        all_logits = torch.cat((out_clean, out_cor), dim=0)
        cross_ent_loss = cross_ent_loss_obj(all_logits, labels.repeat(int(all_logits.shape[0]/args.batch_size)))
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

        # logging
        if args.distributed == True: torch.cuda.synchronize()
        metric_logger.update(cross_ent_loss=cross_ent_loss.item())
        metric_logger.update(cross_ent_loss_poisoned_data=cross_ent_loss_poisoned_data.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser_here = argparse.ArgumentParser(add_help=False)
    parser_here.add_argument("--isolation_ratio", default=0.01, type=float, required=True,
                             help="Amount of data used for unlearning (see ABL)")
    parser_here.add_argument("--test_only", default=False, type=bool, action=argparse.BooleanOptionalAction,
                             help="Whether to load from checkpoint")
    parser = argparse.ArgumentParser('SiT_ensemble', parents=[get_args_parser("finetune"), parser_here])
    args = parser.parse_args()
    args.teacher_dir = os.path.join(args.base_dir, args.teacher_dir_partial)
    args.teacher_dir = args.teacher_dir + "_" + args.poison_method
    args.output_dir = os.path.join(args.base_dir, args.output_dir_partial)
    args.output_dir = args.output_dir + "_" + args.poison_method
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_SiT_supervised(args)
    
