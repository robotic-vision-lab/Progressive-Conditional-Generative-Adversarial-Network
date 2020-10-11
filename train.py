#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:24:29 2019

@author: Samiul Arshad <mohammadsamiul.arshad@mavs.uta.edu>

Training script for PCGAN.

Part of the code was taken from
    -https://github.com/seowok/TreeGAN

"""

import torch
import torch.optim as optim
import json
import logging
import time
from datetime import datetime
import os
from shutil import rmtree
from torch.utils.tensorboard import SummaryWriter

from data.dataset_benchmark import BenchmarkDataset
from model.gan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty
from arguments import Arguments
from utils import find_latest_epoch_and_step, setup_logging, writer_histogram, \
                    one_hot

# class : lable -> {'airplane': 0, 'chair': 1, 'motorcycle': 2, 'sofa': 3, 'table': 4}

class TreeGAN():
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # ------------------------------ Logger ---------------------------- #
        logger = logging.getLogger()
        if not len(logger.handlers):
            setup_logging(self.args.result_root)
        self.log = logging.getLogger(__name__)
        self.log.debug('Using '+str(torch.cuda.device_count()) + ' GPUs!')


        self.prepare_results_dir()
        # Save args in file
        if not os.path.exists(os.path.join(self.args.result_root, 'config.json')):
            with open(os.path.join(self.args.result_root, 'config.json'), mode='w') as f:
                json.dump(self.args.__dict__, f)
        # Set device
        self.args.device = torch.device('cuda:'+str(self.args.gpu) if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.args.device)

        # ------------------------------ Dataset ---------------------------- #
        self.data = BenchmarkDataset(root=args.dataset_path,
                                      npoints=args.point_num,
                                      class_choice=args.class_choice)

        self.dataLoader = torch.utils.data.DataLoader(self.data,
                                                      batch_size=args.batch_size,
                                                      shuffle=True, pin_memory=True,
                                                      num_workers=16, drop_last=True)
        self.log.debug(f'Training Dataset : {len(self.data)} prepared.')

        # ------------------------------ Module ---------------------------- #
        self.G = Generator(args).to(args.device)
        self.G = torch.nn.DataParallel(self.G)
        self.G = self.G.to(self.args.device)
        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.95))

        self.D = Discriminator(args).to(args.device)
        self.D = torch.nn.DataParallel(self.D)
        self.D = self.D.to(self.args.device)
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.95))

        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)

        self.log.debug(f"Network prepared.")

    def prepare_results_dir(self):
        # Clean previous files in case of retrain.
        if self.args.retrain and os.path.isdir(self.args.result_root):
            self.log.warning('Attention! Cleaning results directory in 10 seconds!')
            time.sleep(10)
            rmtree(self.args.result_root)
        # Create appropriate result dirs.
        os.makedirs(self.args.result_root, exist_ok=True)
        os.makedirs(os.path.join(self.args.result_root,
                                 'checkpoints'), exist_ok=True)
        self.args.checkpoint_path = os.path.join(self.args.result_root,
                                                 'checkpoints')
        os.makedirs(os.path.join(self.args.result_root,
                                 'samples'), exist_ok=True)
        self.args.samples_path = os.path.join(self.args.result_root, 'samples')
        os.makedirs(os.path.join(self.args.result_root,
                                 'runs'), exist_ok=True)
        self.args.logs_path = os.path.join(self.args.result_root,
                                 'runs', str(datetime.now()))

    def save_results(self, step, epoch):
        fake_pointclouds = torch.Tensor([])
        for i in range(10): # For batch_size*10 samples
                z = torch.randn(self.args.batch_size, 1, 64).to(self.args.device)
                label = torch.randint(0, len(self.args.class_choice), (self.args.batch_size, 1))
                label_oh = one_hot(label, len(self.args.class_choice))

                with torch.no_grad():
                    sample = self.G([z, label_oh], step).cpu()
                label = label.type(torch.FloatTensor).unsqueeze(1).repeat(1,sample.shape[1],1)
                sample = torch.cat((sample, label), dim=2)

                fake_pointclouds = torch.cat((fake_pointclouds, sample), dim=0)
        path = os.path.join(self.args.samples_path, f'step_{step}_sample_{epoch:05}.pt')
        torch.save(fake_pointclouds, path)
        del fake_pointclouds
        self.save_checkpoints(step, epoch)

    def save_checkpoints(self, step, epoch):
    # ---------------------- Save checkpoint --------------------- #
        torch.save({
                'D_state_dict': self.D.state_dict(),
                'G_state_dict': self.G.state_dict(),
        }, os.path.join(self.args.checkpoint_path,
                        f'step_{step}_epoch_{epoch:05}.pt'))

        self.log.debug(f'Checkpoint is saved.')

    def load_checkpoint(self, step, epoch):
        # Expand model for step. No change needed for first step.
        if step > 0:
            self.change_params_for_step(current=0, target=step, epoch=epoch)

        # Load checkpoints
        path = os.path.join(self.args.checkpoint_path,
                            f'step_{step}_epoch_{epoch:05}.pt')

        self.log.debug(f'Loading Checkpoint: {path}')
        checkpoint = torch.load(path, map_location=self.args.device)

        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.log.debug(f'Checkpoint loaded.')


    def change_params_for_step(self, current, target, epoch=-1):
        self.log.debug(f'Changing Params for step: {target} and epoch: {epoch}')
        for i in range(current, target):
            self.G.module.expand(i+1)

        self.G = self.G.to(self.args.device)
        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.95))

        self.D.module.k = 20 + target*10
        self.D = self.D.to(self.args.device)
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.95))

        # increasing points number & decreasing batch size
        self.data.npoints = self.args.point_num * pow(2,target)
        self.args.batch_size = self.args.batch_size // pow(2,target)

        del self.dataLoader
        self.dataLoader = torch.utils.data.DataLoader(self.data,
                                                      batch_size=self.args.batch_size,
                                                      shuffle=True, pin_memory=False,
                                                      num_workers=16, drop_last=True)

        self.log.debug(f'Current num points: {self.data.npoints} '
                       f'batch size: {self.args.batch_size}')

    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):
        writer = SummaryWriter(self.args.logs_path)
        writer.add_text('args', str(self.args), 0)

        # load most current step and epoch. returns 0 for step, -1 for epoch if nothing was found.
        starting_step, starting_epoch = find_latest_epoch_and_step(self.args.checkpoint_path)
        if not self.args.retrain and starting_step+1 and starting_epoch+1:
            self.load_checkpoint(starting_step, starting_epoch)

        for step in range(starting_step, self.args.total_step):
            for epoch in range(starting_epoch+1, self.args.epochs):
                start_time = time.time()
                G_loss, D_loss = 0., 0.

                for _iter, data in enumerate(self.dataLoader):
                    point, y = data[0], data[1]
                    point = point.to(self.args.device)
                    y_onehot = torch.Tensor(one_hot(y,
                                                    len(self.args.class_choice))).to(self.args.device)

                    # ---------------------- Discriminator -------------------- #
                    for d_iter in range(self.args.D_iter):
                        self.D.zero_grad()

                        z = torch.randn(self.args.batch_size, 1, 64).\
                            to(self.args.device)

                        with torch.no_grad():
                            fake_point = self.G([z, y_onehot], step)

                        D_real = self.D([point, y_onehot])
                        D_realm = D_real.mean()

                        D_fake = self.D([fake_point, y_onehot])
                        D_fakem = D_fake.mean()

                        gp_loss = self.GP(self.D, point.data, fake_point.data, y_onehot)

                        d_loss = -D_realm + D_fakem
                        d_loss_gp = d_loss + gp_loss
                        d_loss_gp.backward()
                        self.optimizerD.step()

                    # ---------------------- Generator ---------------------- #
                    self.G.zero_grad()

                    z = torch.randn(self.args.batch_size, 1, 64).to(self.args.device)

                    fake_point = self.G([z, y_onehot], step)
                    G_fake = self.D([fake_point, y_onehot],)
                    G_fakem = G_fake.mean()

                    g_loss = -G_fakem
                    g_loss.backward()
                    self.optimizerG.step()

                    G_loss += g_loss.item()
                    D_loss += d_loss.item()
                    # --------------------- Visualization -------------------- #
                    self.log.debug(f' Step: {step: 3d}'
                                   f' Epoch/Iter: {epoch:3d} / {_iter:3d} '
                                   f' G_Loss: {g_loss: 7.5f} '
                                   f' D_Loss:  {d_loss: 7.6f} '
                                   f' Time: {(time.time()-start_time):4.2f}s')

                mean_D = D_loss/len(self.dataLoader)
                mean_G = G_loss/len(self.dataLoader)

                self.log.debug(f' Step: {step: 3d}'
                               f' Epoch: {epoch:3d} '
                               f' G_mean {mean_G:7.6f}'
                               f' D_mean: {mean_D:7.6f} '
                               f' Time: {(time.time()-start_time):4.2f}s')

                # ------------------ Summery Writer ----------------- #
                writer.add_scalar('Loss/D', mean_D, self.args.epochs*step+epoch)
                writer.add_scalar('Loss/G', mean_G, self.args.epochs*step+epoch)
                writer_histogram(writer, self.D, self.args.epochs*step+epoch)
                writer_histogram(writer, self.G, self.args.epochs*step+epoch)

                # ---------------- Save Generated Pointcloud --------------- #
                if epoch > 0 and epoch % self.args.saving_freq == 0:
                    self.save_results(step, epoch)

            if step+1 < self.args.total_step:
                # update params for most current step
                self.change_params_for_step(step, step+1)
                starting_epoch = -1

        writer.close()
        del self.data, self.dataLoader

if __name__ == '__main__':
    args = Arguments().parser().parse_args()
    args.result_root = os.path.join(args.result_root, args.experiment)

    model = TreeGAN(args)
    model.run()
