

import argparse
import os
from pathlib import Path

from train_utils import *

from train_ensemble import train_SiT_supervised

if __name__ == '__main__':
    parser_here = argparse.ArgumentParser(add_help=False)
    parser_here.add_argument("--poison_module_depth", required=True, type=int, 
                             help="Depth of poison_module")
    parser = argparse.ArgumentParser('SiT_ensemble', parents=[get_args_parser("finetune"), parser_here])
    args = parser.parse_args()
    args.teacher_dir = os.path.join(args.base_dir, args.teacher_dir_partial)
    args.teacher_dir = args.teacher_dir + "_" + args.poison_method
    args.output_dir = os.path.join(args.base_dir, args.output_dir_partial)
    args.output_dir = args.output_dir + "_" + args.poison_method
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_SiT_supervised(args)
    