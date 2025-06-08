# coding=utf-8
import torch
from datautil.actdata.util import *
from datautil.util import mydataset, Nmax
import numpy as np


class ActList(mydataset):
    def __init__(self, args, dataset, root_dir, people_group, group_num, transform=None, target_transform=None):
        super(ActList, self).__init__(args)
        self.domain_num = 0
        self.dataset = dataset
        self.task = 'cross_people'
        self.transform = transform
        self.target_transform = target_transform
        x, cy, py, sy = accumulate_participant_files(args, root_dir, people_group)
        self.people_group = people_group
        self.position = np.sort(np.unique(sy))
        self.comb_position(x, cy, py, sy)
        self.x = self.x[:, :, np.newaxis, :]
        self.transform = None
        self.x = torch.tensor(self.x).float()
        self.dlabels = np.ones(self.labels.shape) * group_num
    
    
    def comb_position(self, x, cy, py, sy):
        for i, peo in enumerate(self.people_group):
            index = np.where(py == peo)[0]
            tx, tcy, tsy = x[index], cy[index], sy[index]
            for j, sen in enumerate(self.position):
                index = np.where(tsy == sen)[0]
                if j == 0:
                    ttx, ttcy = tx[index], tcy[index]
                else:
                    ttx = np.hstack((ttx, tx[index]))
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x, self.labels = np.vstack(
                    (self.x, ttx)), np.hstack((self.labels, ttcy))

    def set_x(self, x):
        self.x = x

class ActListTest(mydataset):
    def __init__(self, args, x, cy, py, sy, people_group, group_num, transform=None, target_transform=None):
        super(ActListTest, self).__init__(args)
        self.domain_num = 0
        self.task = 'cross_people'
        self.transform = transform
        self.target_transform = target_transform
        self.people_group = people_group
        self.position = np.sort(np.unique(sy))
        self.comb_position(x, cy, py, sy)
        self.x = self.x[:, :, np.newaxis, :]
        self.transform = None
        self.x = torch.tensor(self.x).float()
        self.dlabels = np.ones(self.labels.shape) * \
            (group_num-Nmax(args, group_num))

    def comb_position(self, x, cy, py, sy):
        for i, peo in enumerate(self.people_group):
            index = np.where(py == peo)[0]
            tx, tcy, tsy = x[index], cy[index], sy[index]
            for j, sen in enumerate(self.position):
                index = np.where(tsy == sen)[0]
                if j == 0:
                    ttx, ttcy = tx[index], tcy[index]
                else:
                    ttx = np.hstack((ttx, tx[index]))
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x, self.labels = np.vstack(
                    (self.x, ttx)), np.hstack((self.labels, ttcy))

    def set_x(self, x):
        self.x = x