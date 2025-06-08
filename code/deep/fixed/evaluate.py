# coding=utf-8
import collections
import os
import time
import numpy as np

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_eval_args, init_args
from datautil.getdataloader import get_act_test_dataloader
import pickle as cp



def evaluate(args):

    cms = []
    eval_loaders = get_act_test_dataloader(
        args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    args.domain_num = len(eval_loaders)
    print(args.domain_num)
    algorithm = algorithm_class(args).cuda()
    if args.mode == "cv":
        model_path = os.path.join(args.output, 'model_' + str(args.round) + '.pt')
        print("loading from", model_path)
        algorithm.load((model_path)) 
    else:
        # TODO: implement
        model_path = os.path.join(args.output, 'model_' + str(args.round) + '.pt')
        print("loading from", model_path)
        algorithm.load((model_path)) 
        pass
            
    for loader in eval_loaders:
        print()
        acc, pre, rec, f1 = modelopera.accuracy_metrics(algorithm, loader, None)
        cm = modelopera.confusion_matrix_metrics(algorithm, loader, None, num_classes=args.num_classes)
        print( "accuracy ",acc, "precision ", pre, "recall ", rec, "f1 ", f1)
        cms.append(cm)
    
    return cms

            
if __name__ == '__main__':
    args = get_eval_args()
    init_args(args)
    set_random_seed(args.seed1)

    all_datasets = ['HHAR', 'DSA', 'MHEALTH', 'selfBACK', 'PAMAP2', 'GOTOV', 'C24']
    args.loocv = False
    args.round = 0
    if args.mode == "cv":
        cm_round_filename = os.path.join("results","evaluation_results", args.mode, args.dataset)
        os.makedirs(cm_round_filename, exist_ok=True)
        cms_self = []
        
        tmpp = args.act_people[args.dataset]
        domain_num = len(tmpp)
        if domain_num == 1: # in lab datasets
            num_participants = len(tmpp[0])
            args.domain_num = num_participants
            args.total_rounds = min(10, num_participants)
            if num_participants <= 10:                
                print("Applying L.O.O.CV.")
                args.loocv = True
        else: # This is C24
            args.total_rounds = 1 # just one round with the presplit C24, 100 for training, 51 for testing
            num_participants = len(tmpp[1])
            print("number of participants is", num_participants)
            args.domain_num = num_participants            
            print("C24, only doing one round...")  



        for round in range(args.total_rounds):
            print("ROUND ", round)              
            cms = evaluate(args)
            for cm in cms:
                cms_self.append(cm)
            args.round += 1
        
        
        f = open(os.path.join(cm_round_filename, "self.cms"), 'wb')
        cp.dump(cms_self, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    
    
    elif args.mode == "d2d_test":  
        if args.dataset != 'C24':
            # in 2 in
            test_datasets = [ds for ds in all_datasets if ds != args.dataset and ds != 'C24']
            cm_round_filename = os.path.join("results","evaluation_results", args.mode, args.dataset)
            os.makedirs(cm_round_filename, exist_ok=True)
            cms_in2in = []
            for ds in test_datasets:
                print("Evaluating on", ds)
                args.dataset = ds
                cms = evaluate(args)
                for cm in cms:
                    cms_in2in.append(cm)
            
            f = open(os.path.join(cm_round_filename, "in2in.cms"), 'wb')
            cp.dump(cms_in2in, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()

            #### in 2 out 

            ds = 'C24'
            cms_in2out = []
            print("Evaluating on", ds)
            args.dataset = ds
            cms = evaluate(args)
            for cm in cms:
                cms_in2out.append(cm)

            f = open(os.path.join(cm_round_filename, "in2out.cms"), 'wb')
            cp.dump(cms_in2out, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()
        else:
            # out 2 in
            test_datasets = [ds for ds in all_datasets if ds != args.dataset]
            cm_round_filename = os.path.join("results","evaluation_results", args.mode, args.dataset)
            os.makedirs(cm_round_filename, exist_ok=True)
            cms_out2in = []
            for ds in test_datasets:
                print("Evaluating on", ds)
                args.dataset = ds
                cms = evaluate(args)
                for cm in cms:
                    cms_out2in.append(cm)
            f = open(os.path.join(cm_round_filename, "out2in.cms"), 'wb')
            cp.dump(cms_out2in, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()

            #### out 2 out 
            ds = 'C24'
            cms_out2out =[]
            print("Evaluating on", ds)
            args.dataset = ds
            cms = evaluate(args)
            for cm in cms:
                cms_out2out.append(cm)
            f = open(os.path.join(cm_round_filename, "out2out.cms"), 'wb')
            cp.dump(cms_out2out, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()            
