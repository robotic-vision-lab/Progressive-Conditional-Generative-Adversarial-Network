#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:24:29 2019

@author: Samiul Arshad <mohammadsamiul.arshad@mavs.uta.edu> 

Part of the code was taken from
    -https://github.com/seowok/TreeGAN

"""

import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments for TreeGAN.')

        self._parser.add_argument('--experiment', '-e', type=str, default='test/',
                                  help='Experiment name.')
        # Dataset arguments
        self._parser.add_argument('--dataset_path', '-d', type=str,
                                  default='./datasets/shapenet/point-clouds/',
                                  help='Dataset file path.')
        self._parser.add_argument('--class_choice', '-c', type=str, nargs='+',
                                  default=['chair'],
                                  help='Select one or more class to generate: [Airplane, Chair]')
        self._parser.add_argument('--batch_size', type=int, default=4,
                                  help='Integer value for batch size.')
        self._parser.add_argument('--point_num', type=int, default=1024,
                                  help='Integer value for number of points.')

        # Training arguments
        self._parser.add_argument('--total_step', type=int, default=2,
                                  help='Integer value for step size.')
        self._parser.add_argument('--seed', type=int, default=333,
                                  help='GPU number to use.')
        self._parser.add_argument('--gpu', type=int, default=0,
                                  help='GPU number to use.')
        self._parser.add_argument('--epochs', type=int, default=1,
                                  help='Integer value for epochs.')
        self._parser.add_argument('--retrain', type=bool, default=False,
                                  help='Train again from epoch 0.')
        self._parser.add_argument('--saving_freq', type=int, default=5,
                                  help='Frequency to save checkpoints.')
        self._parser.add_argument('--lr', type=float, default=1e-4,
                                  help='Float value for learning rate.')
        self._parser.add_argument('--result_root', type=str,
                                  default='./results/pcgan/',
                                  help='Root for results.')

        # Network arguments
        self._parser.add_argument('--lambdaGP', type=int, default=10,
                                  help='Lambda for GP term.')
        self._parser.add_argument('--D_iter', type=int, default=1,
                                  help='Number of iterations for discriminator.')
        self._parser.add_argument('--support', type=int, default=10,
                                  help='Support value for TreeGCN loop term.')
        self._parser.add_argument('--DEGREE', type=int, default=[1, 2, 2, 2, 2, 64], nargs='+',
                                  help='Upsample degrees for generator.')
        self._parser.add_argument('--G_FEAT', type=int, default=[128, 128, 256, 256, 128, 128, 6], nargs='+',
                                  help='Features for generator.')

    def parser(self):
        return self._parser
