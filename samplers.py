from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler


# class RandomIdentitySampler(object):
#     """
#     Randomly sample N identities, then for each identity,
#     randomly sample K instances, therefore batch size is N*K.
#
#     Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.
#
#     Args:
#         data_source (Dataset): dataset to sample from.
#         num_instances (int): number of instances per identity.
#     """
#     def __init__(self, data_source, num_instances=4):
#         self.data_source = data_source
#         self.num_instances = num_instances
#         self.index_dic = defaultdict(list)
#         for index, (_, pid, _) in enumerate(data_source):
#             self.index_dic[pid].append(index)
#         self.pids = list(self.index_dic.keys())
#         self.num_identities = len(self.pids)
#
#     def __iter__(self):
#         indices = torch.randperm(self.num_identities)
#         ret = []
#         for i in indices:
#             pid = self.pids[i]
#             t = self.index_dic[pid]
#             replace = False if len(t) >= self.num_instances else True
#             t = np.random.choice(t, size=self.num_instances, replace=replace)
#             ret.extend(t)
#         return iter(ret)
#
#     def __len__(self):
#         return self.num_identities * self.num_instances



class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
        返回数据指数，dataset按照数据指数进行读取
    """

    def __init__(self, train_thermal_label, train_color_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        
        N = np.minimum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N/(batchSize*num_pos))):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)  
            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        # print("index1.shape = {}".format(index1.shape))
        # print("index2.shape = {}".format(index2.shape))
        # print("index1 = {}".format(index1))
        # print("index2 = {}".format(index2))
        # print("N = {}".format(N))
        # print("return iter {}".format(np.arange(len(index1))))
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N  
