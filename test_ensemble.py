"""
This code is for supervised training the ensemble model (student or teacher). 
"""

import os, datetime, time, tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision

from datasets.poisoned_dataset import get_poisoned_dataset
from datasets.datasets_utils import basic_transforms

import utils

from train_utils import *

 
def test_SiT_supervised(args):

    device = torch.device(f'cuda:{args.device}')

    if args.distributed == True:
        utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    # print("git:\n  {}\n".format(utils.get_sha()))
    cudnn.benchmark = False
    cudnn.deterministic = True
    loss_obj = nn.CrossEntropyLoss()
    # prepare dataset for finetuning
    transform = basic_transforms(args, train=False, 
                                 add_ToPILImage=False if (args.poison_method != "ISSBA" or args.test_with_poison == False) else True)
    
    if args.test_with_poison == True:
        dataset = get_poisoned_dataset(method=args.poison_method, dataset_name=args.data_set, train=False, transform=transform, 
                                        trigger_label=args.trigger_label, poisoning_rate=1.0)
        print("Successfully loaded in poisoned dataset")
    else:
        if args.data_set == "CIFAR10":
            dataset = torchvision.datasets.CIFAR10("./datasets/cifar10", train=False, transform=transform) 
        elif args.data_set == "gtsrb":
            dataset = torchvision.datasets.GTSRB("./datasets", split="test", transform=transform)
        elif args.data_set == "tiny_imagenet_200":
            dataset =  torchvision.datasets.DatasetFolder("./datasets/tiny_imagenet_200/val/images", 
                                                          transform=transform, extensions="jpeg", 
                                                          loader=utils.loader)
    match args.data_set:
        case "CIFAR10":
            num_classes = 10
        case "gtsrb":
            num_classes = 43
        case "tiny_imagenet_200":
            num_classes = 200
    
    data_loader = torch.utils.data.DataLoader(dataset,
        batch_size=256, num_workers=4, pin_memory=True, drop_last=False, 
        collate_fn=collate_batch(args.drop_replace, args.drop_align, apply_poison=True))
        # collate_batch only does collating if (self.drop_replace > 0 and self.apply_poison == False)
    print(f"Data loaded: there are {len(dataset)} images.")

    model, _ = return_model(args, student_or_teacher=args.student_or_teacher, num_classes=num_classes)
    
    model = model.to(device)    

    # load in teacher
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint{args.load_epoch:04}.pth" if args.load_epoch is not None else "checkpoint.pth")
    state_dict_model = torch.load(checkpoint_path, map_location=device)
    print(f"---------- Loading from epoch {state_dict_model['epoch']} ----------")
    state_dict_model = state_dict_model[args.student_or_teacher]
    if args.distributed == False:
        state_dict_model = remove_distr(state_dict_model)
    print(model.load_state_dict(state_dict_model, strict=True))

    start_time = time.time()
    with torch.no_grad():
        res = test_one_epoch(model, loss_obj, data_loader, args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))
    if args.write_to_log == True:
        with open(os.path.join(args.output_dir, "test_log.txt"), "a") as fptr:
            if args.student_or_teacher == "student":
                avg_acc_robust, avg_loss_robust, avg_acc_poisoned, avg_loss_poisoned = res
                out_str = "checkpoint_path %s, test_with_poison %s, avg_acc_robust: %.6g, avg_loss_robust: %.6g, avg_acc_poisoned: %.6g, avg_loss_poisoned: %.6g\n" % \
                        (checkpoint_path, args.test_with_poison, avg_acc_robust, avg_loss_robust, avg_acc_poisoned, avg_loss_poisoned)
                fptr.write(str(datetime.datetime.now()) + ":\n")
                fptr.write(out_str)
                print(out_str)
            elif args.student_or_teacher == "teacher":
                avg_acc, avg_loss = res
                out_str = "checkpoint_path %s, test_with_poison %s, avg_acc: %.6g, avg_loss: %.6g\n" % \
                        (checkpoint_path, args.test_with_poison, avg_acc, avg_loss)
                fptr.write(str(datetime.datetime.now()) + ":\n")
                fptr.write(out_str)
                print(out_str)
    return


def test_one_epoch(model, loss_obj, data_loader, args):
    model.eval()
    
    device = torch.device(f'cuda:{args.device}')

    
    if args.student_or_teacher == "student":
        total_examples, correct_examples_robust, correct_examples_poisoned, test_loss_robust, test_loss_poisoned = 0, 0, 0, 0, 0
        for it, (batch, labels) in tqdm.tqdm(enumerate(data_loader)):        
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
    elif args.student_or_teacher == "teacher":
        test_loss, correct_examples, total_examples = 0, 0, 0
        for it, (batch, labels) in tqdm.tqdm(enumerate(data_loader)):        
            batch, labels = batch.to(device), labels.to(device)
            labels = labels.to(device)
            model_in = batch
            cor_out = model(model_in)
            cross_ent_loss = loss_obj(cor_out, labels)
            test_loss += cross_ent_loss.item()
            correct_examples += sum(torch.argmax(cor_out, dim=1, keepdim=False) == labels).item()
            total_examples += len(labels.view(labels.size(0), -1))
        avg_acc = correct_examples / total_examples
        avg_loss = test_loss / len(data_loader)
        return avg_acc, avg_loss
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_file', type=str, help='Where to load in the args file')
    parser.add_argument('--test_with_poison', type=bool, default=False, help='Whether test set has poisoned data or not', 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--load_epoch', type=int, default=None, help="Which epoch's checkpoint to load")
    parser.add_argument('--device', type=int, default=0, help="Which device")
    parser.add_argument('--student_or_teacher', type=str, required=True, choices=["student", "teacher"], help="Defended (student) or poisoned (teacher) model?")
    parser.add_argument('--write_to_log', type=bool, default=False, help='Write to log?', action=argparse.BooleanOptionalAction)
    args_ = parser.parse_args()
    args = load_args(arg_file_path=args_.arg_file)
    args.test_with_poison = args_.test_with_poison
    args.write_to_log = args_.write_to_log
    args.device = args_.device
    args.load_epoch = args_.load_epoch
    args.student_or_teacher = args_.student_or_teacher

    test_SiT_supervised(args)
