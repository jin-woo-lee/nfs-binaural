#!/usr/bin/env python3
import argparse
import os
from train import train
from test import test
import logging
import logging.handlers
from datetime import datetime
from shutil import copyfile
import sys
import traceback

def get_parser():
    parser = argparse.ArgumentParser()
    #------------------------------ 
    # General
    #------------------------------ 
    parser.add_argument('--project', type=str, default='binaural')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--valid_epoch', type=int, default=1)
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--plot_iter', type=int, default=10000)
    parser.add_argument('--dont_plot', action='store_true')
    parser.add_argument('--board_iter', type=int, default=10000)
    parser.add_argument('--total_epochs', type=int, default=16)
    parser.add_argument('--ckpt', type=str, default=None, help='load checkpoint')
    parser.add_argument('--optimizer', default='radam', type=str,
        choices=['sgd', 'adam', 'radam', 'adamw', 'novograd'], help='Optimizer')
    parser.add_argument('--scheduler', default='step', type=str,
        choices=['step', 'multistep','sgdr','lambda_lr','reduce_on_plateau','cosine_annealing'],
        help='Scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--model', type=str, default='nfs')
    parser.add_argument('--ir_c', type=float, default=1.00)
    parser.add_argument('--tf_c', type=float, default=0.10)
    parser.add_argument('--sp_c', type=float, default=1e4)
    parser.add_argument('--gd_c', type=float, default=0.10)
    parser.add_argument('--ld_c', type=float, default=10.0)
    parser.add_argument('--ad_c', type=float, default=0.10)
    parser.add_argument('--dl_c', type=float, default=1.00)
    #------------------------------ 
    # DDP
    #------------------------------ 
    parser.add_argument('--gpus', nargs='+', default=[0,1], help='gpus')
    parser.add_argument('--n_nodes', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--port', default='12345', type=str, help='port')
    #------------------------------ 
    # Loss
    #------------------------------ 
    parser.add_argument('--l2', action='store_true')
    parser.add_argument('--phs', action='store_true')
    parser.add_argument('--mrstft', action='store_true')
    parser.add_argument('--ild', action='store_true')
    parser.set_defaults(l2=True)
    parser.set_defaults(phs=True)
    parser.set_defaults(mrstft=True)
    parser.set_defaults(ild=True)
    #------------------------------ 
    # Network
    #------------------------------ 
    parser.add_argument('--channel', type=int, default=128)
    parser.add_argument('--n_mel', type=int, default=128)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--model_window_ms', type=float, default=200)
    parser.add_argument('--cdim', type=int, default=128)
    parser.add_argument('--wo_ni', action='store_true')
    parser.add_argument('--wo_lff', action='store_true')
    parser.add_argument('--wo_shifter', action='store_true')
    parser.add_argument('--wo_geowarp', action='store_true')
    #------------------------------ 
    # Data
    #------------------------------ 
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lens_sec', type=float, default=0.8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--test_lens_sec',  type=float, default=6.0)
    parser.add_argument('--sr', type=int, default=48000)
    parser.add_argument('--stft_window_ms', type=float, default=40)
    parser.add_argument('--stft_hoplen_ms', type=float, default=10)
    #------------------------------ 
    # etc.
    #------------------------------ 
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--memo', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_num) for gpu_num in args.gpus])
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = args.port

    if args.exp_name is None:
        exp_name = [
            f'{args.model}',
        ]
        if args.wo_ni:
            exp_name += ["woNI"]
        if args.wo_lff:
            exp_name += ["woLFF"]
        if args.wo_geowarp:
            exp_name += ["woGeoWarp"]
        if args.wo_shifter:
            exp_name += ["woShifter"]
        if args.memo is not None:
            exp_name += [args.memo]
        exp_name = '-'.join(exp_name)

    # append datetime to exp_name
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = [exp_name, run_id]
    exp_name = '-'.join(exp_name)
    if args.result_dir is None:
        args.result_dir = exp_name if not args.debug else 'debug'

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel("INFO")
    stream_handler.setFormatter(formatter)

    if args.train:
        # Copy directory sturcture and files
        exclude_dir  = ['results', '__pycache__', 'log', 'wandb']
        exclude_file = ['cfg', 'cmd', 'check_gd.py',]
        exclude_ext  = ['.png', '.jpg', '.pt']
        filepath = []
        for dirpath, dirnames, filenames in os.walk(os.getcwd(), topdown=True):
            if not any(dir in dirpath for dir in exclude_dir):
                filtered_files=[name for name in filenames if (os.path.splitext(name)[-1] not in exclude_ext) and (name not in exclude_file)]
                filepath.append({'dir': dirpath, 'files': filtered_files})

        num_strip = len(os.getcwd())
        for path in filepath:
            dirname = path['dir'][num_strip+1:]
            dirpath2save = os.path.join(f"results/{args.result_dir}", 'codes', dirname)
            os.makedirs(dirpath2save, exist_ok=True)

            for filename in path['files']:
                if 'swp' in filename or 'onnx' in filename:
                    continue
                file2copy = os.path.join(path['dir'], filename)
                filepath2save = os.path.join(dirpath2save, filename)
                copyfile(file2copy, filepath2save)

        if args.resume and args.ckpt==None:
            ckpt_path = f'results/{args.result_dir}/train/ckpt/{args.load_epoch}/{args.model}_{args.load_step}.pt'
            if os.path.exists(ckpt_path):
                args.ckpt = ckpt_path
            else:
                raise FileNotFoundError(
                    "Specify checkpoint by '--ckpt=...'.",
                    "Otherwise provide exact setting for exp_name, load_epoch, load_step.",
                    ckpt_path
                )
        train(args)
    if args.test:
        if args.ckpt==None:
            ckpt_path = f'results/{args.result_dir}/train/ckpt/{args.load_epoch}/{args.model}_{args.load_step}.pt'
            if os.path.exists(ckpt_path):
                args.ckpt = ckpt_path
            else:
                raise FileNotFoundError(
                    f"Specify checkpoint by '--ckpt=...'. "
                    f"Otherwise provide exact setting for exp_name, load_epoch, load_step. {ckpt_path}"
                )
        test(args)


if __name__=='__main__':
    main()

