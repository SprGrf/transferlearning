# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import argparse


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    # if args.domain_num > 1: # C24
    #     for i in range(args.domain_num):
    #         if i not in args.test_envs:
    #             eval_name_dict['train'].append('eval%d_in' % i)
    #             eval_name_dict['valid'].append('eval%d_out' % i)
    #         else:
    #             eval_name_dict['target'].append('eval%d_out' % i)
    #     return eval_name_dict
    # else: 
    #     eval_name_dict['train'].append('train_0')
    #     eval_name_dict['valid'].append('valid_0')
    #     eval_name_dict['target'].append('test_0')
    #     return eval_name_dict
    eval_name_dict['valid'].append('valid_0')
    eval_name_dict['target'].append('test_0')
    for i in range(args.domain_num):
        eval_name_dict['train'].append('train_%d' % i)
    return eval_name_dict
    

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def act_param_init(args):
    # args.act_dataset = ['usc']
    args.act_people = {
        'C24': [[2, 3, 4, 5, 6, 7, 8, 11, 15, 18, 20, 21, 22, 25, 27, 30,
                33, 35, 36, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                54, 56, 57, 59, 60, 61, 62, 64, 67, 68, 69, 71, 72, 73, 76, 77, 79, 80,
                81, 82, 83, 84, 85, 86, 88, 91, 92, 96, 97, 98, 99, 100, 102, 103, 104,
                105, 106, 107, 109, 110, 111, 112, 113, 115, 117, 118, 119, 120, 122,
                125, 128, 130, 131, 132, 133, 134, 135, 139, 141, 142, 143, 144, 145,
                146, 148, 150], 
                [1, 9, 10, 12, 13, 14, 16, 17, 19, 23, 24, 26, 28,
                29, 31, 32, 34, 37, 41, 55, 58, 63, 65, 66, 70, 74, 75, 78, 87, 89, 90,
                93, 94, 95, 101, 108, 114, 116, 121, 123, 124, 126, 127, 129, 136, 137,
                138, 140, 147, 149, 151]],
        'MHEALTH':[[i for i in range(1,11)]],
        'DSA':[[i for i in range(1,9)]],
        'GOTOV':[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                     23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]],
        'HHAR':[[i for i in range(1,10)]],
        'PAMAP2':[[i for i in range(1,9)]],
        'selfBACK':[[26, 27, 28, 29, 
                     30, 31, 33, 34, 36, 39,
                     40, 41, 42, 43, 44, 46, 47, 48, 49,
                     50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                     60, 61, 62, 63]]}
    
    args.special_participants = { 'C24': [[12,47,98,113,131,132],[25,48,127,128,143]],}
    tmp = {'C24': ((3, 1, 125), 7), 'MHEALTH': ((3, 1, 125), 7),
           'GOTOV': ((3, 1, 125), 7), 'HHAR': ((3, 1, 125), 7), 'DSA': ((3, 1, 125), 7),
           'selfBACK': ((3, 1, 125), 7), 'PAMAP2': ((3, 1, 125), 7)}
    args.num_classes, args.input_shape = tmp[args.dataset][1], tmp[args.dataset][0]
    return args


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--mode', type=str, default="cv")
    parser.add_argument('--alpha', type=float,
                        default=0.1, help="DANN dis alpha")
    parser.add_argument('--batch_size', type=int,
                        default=32, help="batch_size")
    parser.add_argument('--beta1', type=float, default=0.5, help="Adam")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=100, help='Checkpoint every N steps')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--class_balanced', type=int, default=0)
    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dsads')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--dis_hidden', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--ldmarginlosstype', type=str, default='avg_top_k',
                        choices=['all_top_k', 'worst_top_k', 'avg_top_k'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--max_epoch', type=int,
                        default=150, help="max iterations")
    parser.add_argument('--mixupalpha', type=float, default=0.2)
    parser.add_argument('--mixup_ld_margin', type=float, default=10000)
    parser.add_argument('--mixupregtype', type=str,
                        default='l-margin', choices=['ld-margin'])
    parser.add_argument('--net', type=str,
                        default='ActNetwork', help="ActNetwork")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed1', type=int, default=10)
    parser.add_argument('--task', type=str,
                        default="cross_people", choices=['cross_people'])
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--output', type=str, default="train_output")
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--wtype', type=str, default='ori',
                        choices=['ori', 'abs', 'fea'])
    args = parser.parse_args()
    return args


def get_eval_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--mode', type=str, default="cv")
    parser.add_argument('--alpha', type=float,
                        default=0.1, help="DANN dis alpha")
    parser.add_argument('--batch_size', type=int,
                        default=32, help="batch_size")
    parser.add_argument('--beta1', type=float, default=0.5, help="Adam")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=100, help='Checkpoint every N steps')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--class_balanced', type=int, default=0)
    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dsads')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--dis_hidden', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--ldmarginlosstype', type=str, default='avg_top_k',
                        choices=['all_top_k', 'worst_top_k', 'avg_top_k'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--max_epoch', type=int,
                        default=150, help="max iterations")
    parser.add_argument('--mixupalpha', type=float, default=0.2)
    parser.add_argument('--mixup_ld_margin', type=float, default=10000)
    parser.add_argument('--mixupregtype', type=str,
                        default='l-margin', choices=['ld-margin'])
    parser.add_argument('--net', type=str,
                        default='ActNetwork', help="ActNetwork")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed1', type=int, default=10)
    parser.add_argument('--task', type=str,
                        default="cross_people", choices=['cross_people'])
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--output', type=str, default="train_output")
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--wtype', type=str, default='ori',
                        choices=['ori', 'abs', 'fea'])
    args = parser.parse_args()
    return args

def init_args(args):
    args.steps_per_epoch = 10000000000
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = act_param_init(args)
    return args
