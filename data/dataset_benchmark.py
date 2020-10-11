#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:24:29 2019

@author: Samiul Arshad <mohammadsamiul.arshad@mavs.uta.edu>

Part of the code was taken from:
     -https://github.com/seowok/TreeGAN

"""
from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np

synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}

category_to_synth_id = {v: k for k, v in synth_id_to_category.items()}
synth_id_to_number = {k: i for i, k in enumerate(synth_id_to_category.keys())}

class BenchmarkDataset(data.Dataset):
    def __init__(self, root, npoints=2500, uniform=False, classification=False, class_choice=[]):
        self.npoints = npoints
        self.root = root
        self.catfile = './data/synsetoffset2category.txt'
        self.cat = {}
        self.uniform = uniform
        self.classification = classification

        if len(class_choice) < 1: self.cat = category_to_synth_id
        else:
            for c in class_choice: self.cat[c] = category_to_synth_id[c]

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            # dir_seg = os.path.join(self.root, self.cat[item], 'points_label')

            # fns = sorted(os.listdir(dir_point))

            for fn in os.listdir(dir_point):
                # token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, fn))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))


        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[fn[0]]
        point_set = np.load(fn[1]).astype(np.float32)

        if self.npoints != point_set.shape[0]:
            choice = np.random.randint(0, point_set.shape[0], size=self.npoints)
            point_set = point_set[choice]

        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.datapath)



