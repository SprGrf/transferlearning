# coding=utf-8
from torch.utils.data import DataLoader
import datautil.actdata.util as actutil
from datautil.util import make_weights_for_balanced_classes, split_trian_val_test
from datautil.mydataloader import InfiniteDataLoader
import datautil.actdata.cross_people as cross_people
import random
import numpy as np
import copy

task_act = {'cross_people': cross_people}

def get_dataloader(args, trdatalist, tedatalist):
    in_splits, out_splits = [], []
    for tr in trdatalist:
        if args.class_balanced:
            in_weights = make_weights_for_balanced_classes(tr)
        else:
            in_weights = None
        in_splits.append((tr, in_weights))
    for te in tedatalist:
        if args.class_balanced:
            out_weights = make_weights_for_balanced_classes(te)
        else:
            out_weights = None
        out_splits.append((te, out_weights))

    print("in splits", len(in_splits))
    print("out splits", len(out_splits))
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)]

    print("train_loaders is", train_loaders)
    tr_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for i, (env, env_weights) in enumerate(in_splits)]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for i, (env, env_weights) in enumerate(in_splits + out_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    print("length of eval loaders is", len(eval_loaders))
    return train_loaders, tr_loaders, eval_loaders, in_splits, out_splits, eval_weights


def get_act_dataloader(args):
    train_datasetlist = []
    eval_datasetlist = []
    pcross_act = task_act[args.task]
    in_names, out_names = [], []
    trl = []
    tmpp = copy.deepcopy(args.act_people[args.dataset])    
    
    if args.dataset in args.special_participants.keys():
        special_participant_list = args.special_participants[args.dataset]
    else:
        special_participant_list = []
    domain_num = len(tmpp)
    # First find the training, validation and test users, so we can split them in domains
    if domain_num == 1:
        train_users, valid_users, test_users = select_participants(tmpp[0], special_participant_list, 0.8, test=True, loocv=args.loocv, round=args.round)
        tdata = pcross_act.ActList(
                args, args.dataset, args.data_dir, valid_users, 0, transform=actutil.act_test())
        out_names.append('valid_0')
        eval_datasetlist.append(tdata)
        trl.append(tdata)
        
        tdata = pcross_act.ActList(
                args, args.dataset, args.data_dir, test_users, 0, transform=actutil.act_test())
        eval_datasetlist.append(tdata)
        out_names.append('test_0')
    else: 
        for i, item in enumerate(tmpp):
            if i in args.test_envs:
                test_users = item

                tdata = pcross_act.ActList(
                    args, args.dataset, args.data_dir, test_users, i, transform=actutil.act_test())
                eval_datasetlist.append(tdata)
                out_names.append('test_0')
                trl.append(tdata)
            else:
                train_users, valid_users, _ = select_participants(item, special_participant_list, 0.8)


                tdata = pcross_act.ActList(
                        args, args.dataset, args.data_dir, valid_users, i, transform=actutil.act_test())
                out_names.append('valid_0')
                eval_datasetlist.append(tdata)
                trl.append(tdata)


    # Training users will be split in domains, a list of lists, so we are compatible with the original FIXED architecture
    users_per_domain = 3 
    domains = [train_users[i:i + users_per_domain] for i in range(0, len(train_users), users_per_domain)]
    print("new training domains are", domains)
    
    
    args.domain_num = len(domains)
    print("number of domains are", args.domain_num)

    for i, item in enumerate(domains):
        print("item is", item)
        print("i is", i)
        tdata = pcross_act.ActList(
            args, args.dataset, args.data_dir, item, i, transform=actutil.act_train())
        in_names.append('train_%d' % (i))
        train_datasetlist.append(tdata)
        trl.append(tdata)

    eval_loader_names = in_names
    eval_loader_names.extend(out_names)
    train_loaders, tr_loaders, eval_loaders, in_splits, out_splits, eval_weights = get_dataloader(
        args, train_datasetlist, eval_datasetlist)
    return train_loaders, tr_loaders, eval_loaders, in_splits, out_splits, eval_loader_names, eval_weights, trl


def get_act_test_dataloader(args):
    eval_datasetlist = []
    pcross_act = task_act[args.task]
    out_names = []
    tmpp = copy.deepcopy(args.act_people[args.dataset])    

    if args.dataset in args.special_participants.keys():
        special_participant_list = args.special_participants[args.dataset]
    else:
        special_participant_list = []
    if args.mode == 'cv':    
        if len(tmpp) == 1: # in lab datasets, need to select the participant
            _, _, test_users = select_participants(tmpp[0], special_participant_list, 0.8, test=True, loocv=args.loocv, round=args.round)
        else: # C24
            test_users = tmpp[1]
    else: # d2d
        if len(tmpp) == 1: # in lab datasets, all participants
            test_users = tmpp[0]
        else: # C24
            test_users = tmpp[1]

    print("test users are", test_users)
    for i, item in enumerate(test_users):
        # print("item is", item)
        tdata = pcross_act.ActList(
                args, args.dataset, args.data_dir, [item], i, transform=actutil.act_test())
        eval_datasetlist.append(tdata)

    eval_loaders = [DataLoader(
        dataset=te,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for te in eval_datasetlist]
    
    return eval_loaders


def select_participants(users_array, special_participant_list, training_rate, test=False, loocv=False, round=None):
    if not 0 <= training_rate <= 1:
        raise ValueError("Percentage must be between 0 and 1.")

    special_sets = [set(special) for special in special_participant_list]

    if not loocv:
        while True:

            users_list = users_array
            random.shuffle(users_list)
            users_array = np.array(users_list)

            if test:
                num_train = int(len(users_array) * training_rate)
                num_remaining = len(users_array) - num_train
                num_val = num_remaining // 2
                train_participants = users_array[:num_train]
                val_participants = users_array[num_train:num_train + num_val] 
                test_participants = users_array[num_train + num_val:]
        
                if (all(special_set.intersection(train_participants) for special_set in special_sets) and
                    all(special_set.intersection(val_participants) for special_set in special_sets) and
                    all(special_set.intersection(test_participants) for special_set in special_sets)):
                    print("train users are", train_participants)
                    print("validation users are", val_participants)
                    print("test users are", test_participants)
                    return train_participants, val_participants, test_participants
                print("Reshuffling as one or more groups lack special participants...")
            else:
                num_train = int(len(users_array) * training_rate)
                train_participants = users_array[:num_train]
                val_participants = users_array[num_train:]
                if (all(special_set.intersection(train_participants) for special_set in special_sets) and
                    all(special_set.intersection(val_participants) for special_set in special_sets)):
                    print("train users are", sorted(train_participants))
                    print("validation users are", val_participants)
                    print("number of validation users", len(val_participants))

                    return train_participants, val_participants, []
                print("Reshuffling as one or more groups lack special participants...")      
    else:
        if round == 0:
            reordered_users_array = users_array 
        else:
            reordered_users_array = list(users_array)[-round:] + list(users_array)[:-round] 
        train_participants = reordered_users_array[:-2]
        val_participants = [reordered_users_array[-2]]
        test_participants = [reordered_users_array[-1]]

        print("train users are", sorted(train_participants))
        print("validation users are", sorted(val_participants))
        print("test users are", sorted(test_participants))
        return train_participants, val_participants, test_participants

