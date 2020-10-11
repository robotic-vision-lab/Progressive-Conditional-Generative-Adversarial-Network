#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:24:29 2019

@author: Samiul Arshad <mohammadsamiul.arshad@mavs.uta.edu> 

"""

import logging
import re
import os
import torch

def one_hot(y, num_class):
    '''
    Converts class label to one hot representation

    Parameters
    ----------
    cls : Class labels in int

    num_class : Total Number of class

    Returns
    -------
    one_hot_cls : Class labels in one hot representation

    '''
    batch = y.shape[0]

    y_onehot = torch.zeros(batch, num_class)
    y_onehot.scatter_(1,y,1)

    return y_onehot

def find_latest_step(path):
    '''
        Finds latest step given a path
    '''
    # Files in format 'step_d_epoch_dddd.pt'
    step_regex = re.compile(r'^step_(?P<n_step>\d+)_epoch_\d+\.pt$')
    steps_completed = []
    for f in os.listdir(path):
        m = step_regex.match(f)
        if m:
            steps_completed.append(int(m.group('n_step')))
    return max(steps_completed) if steps_completed else 0

def find_latest_epoch(path, step):
    '''
        Finds latest epoch given a path and step
    '''
    # Files in format 'step_d_epoch_dddd.pt'
    epoch_regex = re.compile(r'^step_'+str(step)+'_epoch_(?P<n_epoch>\d+)\.pt$')
    epochs_completed = []
    for f in os.listdir(path):
        m = epoch_regex.match(f)
        if m:
            epochs_completed.append(int(m.group('n_epoch')))
    return max(epochs_completed) if epochs_completed else -1

def find_latest_epoch_and_step(path):
    step = find_latest_step(path)
    epoch = find_latest_epoch(path, step)
    return step, epoch

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)

    logpath = os.path.join(log_dir, 'log.txt')
    filemode = 'a' if os.path.exists(logpath) else 'w'
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=logpath,
                        filemode=filemode)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

def writer_histogram(writer, model, epoch):
    for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(model.__class__.__name__+'/'+tag, value.data.cpu().numpy(), epoch)
            try:
                writer.add_histogram(model.__class__.__name__+tag+'/grad', value.grad.cpu().numpy(), epoch)
            except:
                continue


