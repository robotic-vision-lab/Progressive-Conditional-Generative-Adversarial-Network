#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:24:29 2019

@author: Samiul Arshad <mohammadsamiul.arshad@mavs.uta.edu>

Part of the code was taken from
    -https://github.com/seowok/TreeGAN
    -https://github.com/WangYueFt/dgcnn/tree/master/pytorch

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gcn import TreeGCN
from copy import deepcopy

class Discriminator(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Discriminator, self).__init__()
        self.args = args
        self.k = 20

        lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.conv = self.convModule(act=lrelu)

        self.linear = nn.ModuleList([nn.Sequential(nn.Linear(512*2+64, 512, bias=False),
                                                   lrelu),
                                     nn.Sequential(nn.Linear(256*2, 256, bias=False),
                                                   lrelu),
                                     nn.Linear(256, 1)])

        self.fcLabel = nn.ModuleList([nn.Sequential(nn.Linear(len(args.class_choice), 16, bias=True),
                                                   lrelu),
                                     nn.Sequential(nn.Linear(16, 64, bias=True),
                                                   lrelu)])
    def convModule(self, act):
        conv = nn.ModuleList([nn.Sequential(nn.Conv2d(6*2, 64, kernel_size=1, bias=False),
                                            act),
                              nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                            act),
                              nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                            act),
                              nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                            act),
                              nn.Sequential(nn.Conv1d(512, 512*2, kernel_size=1, bias=False),
                                            act)])
        return conv

    def convOp(self, conv, x, k):
        batch_size = x.size(0)
        feat = []

        for j in range(len(conv)):
            if j == 0: # input layer
                x = self.get_graph_feature(x)
                x = conv[j](x)
                feat.append(x.max(dim=-1, keepdim=False)[0])

            elif j == len(conv)-1: # last layer
                x = torch.cat(feat, dim=1)
                x = conv[j](x)

            else:
                x = self.get_graph_feature(feat[-1])
                x = conv[j](x)
                feat.append(x.max(dim=-1, keepdim=False)[0])

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        return x

    def knn(self, x):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]   # (batch_size, num_points, k)
        return idx


    def get_graph_feature(self, x, idx=None, feature_only=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = self.knn(x)   # (batch_size, num_points, k)
        device = torch.device('cuda')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)

        if feature_only: return feature

        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
        return feature

    def forward(self, data):
        x, y = data[0], data[1]
        x = x.transpose(2,1)

        # ---------- label feature -----------#
        for fc in self.fcLabel:
            y = fc(y)

        # ---------- point feature -----------#
        nx = self.convOp(self.conv, x, self.k)

        # ---------- total aggregation -----------#
        x = torch.cat((nx,y),dim=1)

        for i in range(len(self.linear)):
            x = self.linear[i](x)

        return x

class Generator(nn.Module):
    def __init__(self, args):
        self.args = args
        self.layer_num = len(args.G_FEAT)-1
        assert self.layer_num == len(args.DEGREE), \
            "Number of features should be one more than number of degrees."

        self.pointcloud = None
        super(Generator, self).__init__()

        self.vertex_num = 1
        self.gcn = nn.Sequential()


        for inx in range(self.layer_num-1):
            self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(args.batch_size, inx, args.G_FEAT, args.DEGREE[inx],
                                            support=args.support, node=self.vertex_num, upsample=True, activation=True))
            self.vertex_num = int(self.vertex_num * args.DEGREE[inx])

        self.leaf = nn.ModuleDict([['current', TreeGCN(args.batch_size, self.layer_num-1, args.G_FEAT, args.DEGREE[-1],
                                    support=args.support, node=self.vertex_num, upsample=True, activation=False)]])

        lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.fcLabel = nn.ModuleList([nn.Sequential(nn.Linear(len(args.class_choice), 16, bias=True),
                                                   lrelu),
                                     nn.Sequential(nn.Linear(16, 64, bias=True),
                                                   lrelu)])

    def forward(self, data, step):
        tree, y = data[0], data[1]

        # label feature
        for fc in self.fcLabel:
            y = fc(y)
        y = y.unsqueeze(1)
        tree = [torch.cat((tree,y), dim=2)]
        feat = self.gcn(tree)
        feat = self.leaf['current'](feat)

        self.pointcloud = feat[-1]
        return self.pointcloud

    def expand(self, step):
        self.gcn.add_module('TreeGCN_'+str(self.layer_num+step-2), deepcopy(self.gcn[-1]))
        self.gcn[-1].expand(self.vertex_num)
        self.vertex_num = int(self.vertex_num * 2)

        self.leaf['pre'] = deepcopy(self.leaf['current'])
        self.leaf['current'].expand(self.vertex_num)

    def getPointcloud(self):
        return self.pointcloud[-1]
