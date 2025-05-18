"""
This code is for supervised training the ensemble student model. 
"""
import typing, argparse, os, sys, datetime, time, math, json, torch, torchvision
from pathlib import Path

import pandas as pd

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from datasets import datasets_utils
from datasets.poisoned_dataset import get_poisoned_dataset, PoisonedData

from different_arch.full_pipeline import FullPipeline_Ensemble

import utils

from train_utils import *


PRESENT_KEYS = {"clean_crops", "corrupted_crops", "no_crop_imgs", "labels"}
poison_loss_list = []


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

    basic_transforms = datasets_utils.basic_transforms(args, train=False, 
                                 add_ToPILImage=False)
    if args.data_set == "CIFAR10":
        clean_test_set = torchvision.datasets.CIFAR10("./datasets/cifar10", train=False, transform=basic_transforms) 
    elif args.data_set == "gtsrb":
        clean_test_set = torchvision.datasets.GTSRB("./datasets", split="test", transform=basic_transforms)
    elif args.data_set == "tiny_imagenet_200":
        clean_test_set =  torchvision.datasets.DatasetFolder("./datasets/tiny_imagenet_200/val/images", 
                                                        transform=basic_transforms, extensions="jpeg", 
                                                        loader=utils.loader)
    clean_test_loader = torch.utils.data.DataLoader(clean_test_set,
        batch_size=256, num_workers=4, pin_memory=True, drop_last=False, 
        collate_fn=collate_batch(args.drop_replace, args.drop_align, apply_poison=True))

    basic_transforms = datasets_utils.basic_transforms(args, train=False, 
                                 add_ToPILImage=False if args.poison_method != "ISSBA" else True)
    poison_test_set = get_poisoned_dataset(method=args.poison_method, dataset_name=args.data_set, train=False, transform=basic_transforms, 
                                        trigger_label=args.trigger_label, poisoning_rate=1.0)
    poison_test_loader = torch.utils.data.DataLoader(poison_test_set,
        batch_size=256, num_workers=4, pin_memory=True, drop_last=False, 
        collate_fn=collate_batch(args.drop_replace, args.drop_align, apply_poison=True))
    
    
    student, _ = return_model(args, student_or_teacher="student", num_classes=num_classes)
    
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
    # optimizer_prefinetuning = torch.optim.AdamW(utils.get_params_groups(student, substr_to_exclude={"robust_module"}))  # prefinetuning
    optimizer_train = torch.optim.AdamW(utils.get_params_groups(student, substr_to_exclude={"poison_module"}))  # finetuning
    optimizer_poison = torch.optim.AdamW(utils.get_params_groups(student, substr_to_exclude={"poison_module"}))  # poisoned data

    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # init schedulers 
    # lr_schedule_prefinetuning = utils.cosine_scheduler(
    #     base_value=args.prefinetuning_lr * (args.batch_size * utils.get_world_size()) / 256., 
    #     final_value=args.prefinetuning_lr * (args.batch_size * utils.get_world_size()) / 256., 
    #     epochs=args.prefinetuning_epochs, niter_per_ep=len(data_loader), 
    #     warmup_epochs=0 if args.data_set == "gtsrb" else 1)
    lr_schedule_train = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256., 
        args.min_lr, args.finetuning_epochs, len(data_loader), warmup_epochs=args.warmup_epochs)

    wd_schedule = utils.cosine_scheduler(args.weight_decay,
        args.weight_decay_end, args.finetuning_epochs, len(data_loader))

    # load in student's robust_module
    state_dict_student = torch.load("checkpoints/SiT_Small_ImageNet_ViT_student.pth",
                                    map_location=device)
    if args.distributed == False:
        state_dict_student = remove_distr(state_dict_student)
    print(f"student.robust_module.load_state_dict, from path = checkpoints/SiT_Small_ImageNet_ViT_student.pth")
    msg_robust = student.robust_module.load_state_dict(state_dict_student, strict=False)
    print(msg_robust)
    
    poi_path = f"checkpoints/ensemble_all_defended/{args.data_set}/ensemble_{args.poison_method}/checkpoint.pth"
    print(f"student.poison_module.load_state_dict, from path = {poi_path}")
    state_dict_poi = torch.load(poi_path, map_location=device)["student"]
    if args.distributed == False:
        state_dict_poi = remove_distr(state_dict_poi)
    state_dict_poi = {k[14:]: v for k, v in state_dict_poi.items() if "poison_module" in k}
    msg_poison = student.poison_module.load_state_dict(state_dict_poi, strict=True)
    print(msg_poison)
    assert args.prefinetuning_epochs == 0
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
        
        if to_restore['epoch'] > 0:  # next epoch to start with is finetuning AND has already trained for at least one epoch (i.e., training optimizer isnt brand new)
            optimizer_train.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["optimizer"])
        else:
            pass  # no need to load in any optimizer, since prefinetuning just finished and finetuning hasn't started
        print(f"to_restore['epoch'] == {to_restore['epoch']}")
        
        state_dict_student = torch.load(checkpoint_path, map_location=device)["student"]
        if args.distributed == False:
            state_dict_student = remove_distr(state_dict_student)
        print(f"Loading student from {checkpoint_path} since args.restart_from_checkpoint == True...")
        print(student.load_state_dict(state_dict_student, strict=True))
    start_time = time.time()
    
    print(f"(pre)finetuning ... starting from epoch {to_restore['epoch']}")
    results = pd.DataFrame(columns=["epoch", "avg_acc_robust", "avg_loss_robust", "avg_acc_poisoned", "avg_loss_poisoned", "test_with_poison"])
    for epoch in range(to_restore['epoch'], args.finetuning_epochs):
        if args.distributed == True:
            data_loader.sampler.set_epoch(epoch)
        # Training
        # if epoch < 0:
            # optimizer, lr_schedule, mode = optimizer_prefinetuning, lr_schedule_prefinetuning, "prefinetuning"
        # else:
        optimizer, lr_schedule, mode = optimizer_train, lr_schedule_train, "finetuning"
        train_stats = train_one_epoch(student, cross_ent_loss_obj, data_loader, optimizer, None if epoch < 0 else optimizer_poison, 
                                      lr_schedule, wd_schedule, epoch, fp16_scaler, args, 
                                      PoisonedData(batch_size=args.batch_size, device=device, present_keys=PRESENT_KEYS), 
                                      mode=mode, total_epochs=args.finetuning_epochs, num_classes=num_classes)
        # logs
        student_state_dict = student.state_dict()
        if args.distributed == False:
            student_state_dict = add_distr(student_state_dict)
        save_dict = {'student': student_state_dict, 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'args': args}
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        
        with torch.no_grad():
            results.loc[len(results)] = [epoch, *test_one_epoch(student, cross_ent_loss_obj, clean_test_loader, args), False]
            results.loc[len(results)] = [epoch, *test_one_epoch(student, cross_ent_loss_obj, poison_test_loader, args), True]
        results.to_csv(os.path.join(args.output_dir, "results.txt"))

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

    write_weight_output_ = True
    sample_output_dir = os.path.join(args.output_dir, 'sample_output')
    Path(sample_output_dir).mkdir(parents=True, exist_ok=True)

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
                scale_factor = args.scale_factor
                param_group["lr"] = lr_schedule[it] * scale_factor
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[it]
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
        
        def _student_forward(inp, inp_no_crop, 
                             is_corrupt:bool, write_weight_output_:bool, mode:str, get_poisoned_inp:bool):
            out_ = student(inp, confidence_threshold=args.confidence_thresh, x_no_crop=inp_no_crop, 
                           mode=mode)
            poison_out, robust_out, weight = out_["poison_out"], out_["robust_out"], out_["weight"]
            temp_d = {True: ":weight_corru:", False: ":weight_clean:"}
            if write_weight_output_ == True and utils.is_main_process():
                with open(os.path.join(sample_output_dir, "weight_output.txt"), "a") as fptr:
                    if weight is not None:
                        fptr.write(str(epoch) + temp_d[is_corrupt] + str(",".join("%.4f" % x for x in weight.detach().cpu().numpy().flatten()[0:10])) + "\n")
                    else:
                        fptr.write(str(epoch) + temp_d[is_corrupt] + "None...\n")
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
                                True, write_weight_output_, mode, get_poisoned_inp=True)
        out_cor, poisoned_inp = _ret["logits"], _ret["poisoned_inp"]
        
        student_in_clean = torch.cat(clean_crops[0:])
        _ret = _student_forward(student_in_clean, student_in_no_crop,
                                False, write_weight_output_, mode, get_poisoned_inp=False)
        out_clean = _ret["logits"]

        write_weight_output_ = False

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
        if args.distributed == True: torch.cuda.synchronize()
        metric_logger.update(cross_ent_loss=cross_ent_loss.item())
        # metric_logger.update(aux_loss_clean=aux_loss_clean.item())
        # metric_logger.update(aux_loss_cor=aux_loss_cor.item())
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


def test_one_epoch(model, loss_obj, data_loader, args):
    model.eval()
    
    device = torch.device(f'cuda:{args.device}')
    print("testing...")
    
    total_examples, correct_examples_robust, correct_examples_poisoned, test_loss_robust, test_loss_poisoned = 0, 0, 0, 0, 0
    for it, (batch, labels) in enumerate(data_loader):        
        batch, labels = batch.to(device), labels.to(device)
        labels = labels.to(device)
        model_in = batch
        
        cor_out = model(model_in, mode="finetuning", confidence_threshold=args.confidence_thresh)

        logits_robust = cor_out["robust_out"]
        logits_poisoned = cor_out["poison_out"]

        cross_ent_loss_robust = loss_obj(logits_robust, labels)
        cross_ent_loss_poisoned = loss_obj(logits_poisoned, labels)

        test_loss_robust += cross_ent_loss_robust.item()
        test_loss_poisoned += cross_ent_loss_poisoned.item()

        correct_examples_robust += sum(torch.argmax(logits_robust, dim=1, keepdim=False) == labels).item()
        correct_examples_poisoned += sum(torch.argmax(logits_poisoned, dim=1, keepdim=False) == labels).item()
        total_examples += len(labels.view(labels.size(0), -1))
    avg_acc_robust = correct_examples_robust / total_examples
    avg_acc_poisoned = correct_examples_poisoned / total_examples
    avg_loss_robust = test_loss_robust / len(data_loader)
    avg_loss_poisoned = test_loss_poisoned / len(data_loader)
    return avg_acc_robust, avg_loss_robust, avg_acc_poisoned, avg_loss_poisoned
    

if __name__ == '__main__':
    parser_here = argparse.ArgumentParser(add_help=False)
    parser_here.add_argument("--scale_factor", required=True, type=float,
                             help="scale factor on learning rate")
    parser = argparse.ArgumentParser('SiT_ensemble', parents=[get_args_parser("finetune"), parser_here])
    args = parser.parse_args()
    args.teacher_dir = os.path.join(args.base_dir, args.teacher_dir_partial)
    args.teacher_dir = args.teacher_dir + "_" + args.poison_method
    args.output_dir = os.path.join(args.base_dir, args.output_dir_partial)
    args.output_dir = args.output_dir + "_" + args.poison_method
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_SiT_supervised(args)
