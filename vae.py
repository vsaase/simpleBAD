# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import numpy as np
import torch
from torch.utils import data
from torch import nn, optim
from util import create_train_idxs
import torch.distributions as dist
import random
from mri_dataloaders import Dataset3dH5
import pandas as pd
import nibabel as nb
from toh5 import ztransform
from tqdm import tqdm
import os
import h5py, hdf5plugin
from inversecovariance import calc_icov

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, dotransfer=True, residualchans=9):
        super().__init__()
        self.dotransfer = dotransfer
        self.in_channels = in_channels
        self.residualchans = residualchans
        self.conv = nn.Conv3d(in_channels, out_channels, 4, stride=2, padding=1)
        
        if self.dotransfer:
            self.transfer = nn.LeakyReLU(0.2)
        if self.residualchans>0:
            assert(residualchans <= in_channels)
            self.downsample = nn.Upsample(scale_factor=0.5, mode="trilinear", align_corners=False)
        self.conv.weight.data *= 0.1
        self.conv.bias.data *= 0.1

    def forward(self, x):
        if self.residualchans>0:
            xin_down = self.downsample(x[:,:self.residualchans])
        x = self.conv(x)
        if self.dotransfer:
            x = self.transfer(x)
        if self.residualchans>0:
            x = torch.cat([x[:,:self.residualchans] + xin_down, x[:,self.residualchans:]], dim=1)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, dotransfer=True, convupsampling=True, residualchans=9):
        super().__init__()
        self.dotransfer = dotransfer
        self.convupsampling = convupsampling
        self.residualchans = residualchans
        if self.convupsampling:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, 4, stride=2, padding=1) 
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, 1, stride=1, padding=0) 

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        if self.dotransfer:
            self.transfer = nn.LeakyReLU(0.2)
        self.conv.weight.data *= 0.1
        self.conv.bias.data *= 0.1

    def forward(self, x):
        if self.residualchans>0:
            xin_up = self.upsample(x[:,:self.residualchans])
        x = self.conv(x)
        if not self.convupsampling:
            x = self.upsample(x)
        if self.dotransfer:
            x = self.transfer(x)
        if self.residualchans>0:
            x = torch.cat([x[:,:self.residualchans] + xin_up, x[:,self.residualchans:]], dim=1)
        return x

class ResBlock(nn.Module):
    def __init__(self, n_channels, use_gn=False, n_groups=8, n_blocks=1):
        super().__init__()
        self.use_gn=use_gn
        self.n_blocks = n_blocks
        if self.use_gn:
            self.gn = [nn.GroupNorm(min(n_groups, n_channels), n_channels) for i in range(n_blocks)]
            self.gn = nn.ModuleList(self.gn)
        self.conv = [nn.Conv3d(n_channels, n_channels, 3, padding=1) for i in range(n_blocks)]
        self.conv = nn.ModuleList(self.conv)
        self.transfer = nn.LeakyReLU(0.2)
        for conv in self.conv:
            conv.weight.data *= 0.1
            conv.bias.data *= 0.1
    
    def forward(self, x):
        if self.use_gn:
            x1 = self.gn[0](x)
            x1 = self.conv[0](x1)
        else:
            x1 = self.conv[0](x)
        x1 = self.transfer(x1)
        for i in range(1,self.n_blocks):
            if self.use_gn:
                x1 = self.gn[i](x1)
            x1 = self.conv[i](x1)
        x1 = self.transfer(x1) + x
        return x

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, use_resblocks = False, res_n_blocks=2, n_downsample=4, maxchannel=128, zchannel=32, variational=True):
        super().__init__()
        self.use_resblocks = use_resblocks
        self.maxchannel=maxchannel
        self.n_downsample = n_downsample
        self.z_dim = z_dim
        self.zchannel = zchannel
        self.transfer = nn.LeakyReLU(0.2)
        self.variational = variational
        if self.use_resblocks:
            self.rb = [ResBlock(min(maxchannel,32*2**i), n_blocks=res_n_blocks) for i in range(n_downsample-1)]
            self.rb += [ResBlock(zchannel, n_blocks=res_n_blocks)]
            self.rb = nn.ModuleList(self.rb)

        self.l = [DownSample(9, 32)]
        self.l += [DownSample(min(maxchannel,32*2**i), min(maxchannel,32*2**(i+1))) for i in range(n_downsample-2)]
        self.l += [DownSample(min(maxchannel,32*2**(n_downsample-2)), zchannel)]
        self.l = nn.ModuleList(self.l)
        if self.z_dim>0:
            self.transfer = nn.LeakyReLU(0.2)
            self.d_loc = nn.Linear(zchannel*192*224*192//(2**(n_downsample*3)), z_dim)
            if self.variational:
                self.d_scale = nn.Linear(zchannel*192*224*192//(2**(n_downsample*3)), z_dim)
        else:
            self.loc = nn.Conv3d(zchannel,zchannel,1)
            if self.variational:
                self.scale = nn.Conv3d(zchannel,zchannel,1)
            self.loc.weight.data *= 0.1
            self.loc.bias.data *= 0.1

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        for i in range(self.n_downsample):
            x = self.l[i](x)
            if self.use_resblocks:
                x = self.rb[i](x)
        if self.z_dim>0:
            x = x.view(-1,self.zchannel*192*224*192//(2**(self.n_downsample*3)))
            # print(x_loc.shape)

            if self.variational:
                z_logvar = self.transfer(self.d_scale(x))
            z_loc = self.transfer(self.d_loc(x))
        else:
            if self.variational:
                z_logvar = self.transfer(self.scale(x))
            z_loc = self.transfer(self.loc(x)) + x
        
        if self.variational:
            return z_loc, z_logvar
        else:
            return z_loc


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, use_resblocks=False, res_n_blocks=1, n_downsample=4, maxchannel=128, zchannel=32):
        super().__init__()
        self.use_resblocks = use_resblocks
        self.maxchannel=maxchannel
        self.n_downsample = n_downsample
        self.z_dim=z_dim
        self.zchannel = zchannel
        self.transfer = nn.LeakyReLU(0.2)
        if self.use_resblocks:
            self.rb = [ResBlock(min(maxchannel,32*2**i), n_blocks=res_n_blocks) for i in range(n_downsample-1)]
            self.rb += [ResBlock(zchannel, n_blocks=res_n_blocks)]
            self.rb = nn.ModuleList(self.rb)

        self.l = [UpSample(32, 9, dotransfer=False)] #final conv without transfer!
        self.l += [UpSample(min(maxchannel,32*2**(i+1)), min(maxchannel,32*2**i)) for i in range(n_downsample-2)]
        self.l += [UpSample(zchannel, min(maxchannel,32*2**(n_downsample-2)))]
        self.l = nn.ModuleList(self.l)
        if z_dim>0:
            self.d = nn.Linear(z_dim, zchannel*192*224*192//(2**(self.n_downsample*3)))
            self.d.weight.data *= 0.1
            self.d.bias.data *= 0.1

        # setup the non-linearities
        self.transfer = nn.LeakyReLU(0.2)

    def forward(self, z):
        if self.z_dim>0:
            x = self.transfer(self.d(z))
            x = x.view(-1,self.zchannel,192//(2**self.n_downsample),224//(2**self.n_downsample),192//(2**self.n_downsample))
        else:
            x = z
        for i in range(self.n_downsample-1,-1,-1):
            if self.use_resblocks:
                x = self.rb[i](x)
            x = self.l[i](x)
        return x

# define a PyTorch module for the VAE
class VAE(nn.Module):

    def __init__(self, z_dim=4096, use_cuda=False, use_resblocks=False, enc_res_n_blocks=2, 
                dec_res_n_blocks=1, n_downsample=5, maxchannel=128, zchannel=16, variational=True):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, use_resblocks, res_n_blocks=enc_res_n_blocks, n_downsample=n_downsample, maxchannel=maxchannel, zchannel=zchannel, variational=variational)
        self.decoder = Decoder(z_dim, use_resblocks, res_n_blocks=dec_res_n_blocks, n_downsample=n_downsample, maxchannel=maxchannel, zchannel=zchannel)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.variational = variational

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu,logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        if self.variational:
            mu, logvar = self.encode(x)
            z = self.reparametrize(mu, logvar)
            return self.decode(z), mu, logvar
        else:
            return self.decoder(self.encoder(x))

def loss_function(recon_x, x, mu, logvar, mse=True):
    if mse:
        pixelloss = torch.mean((recon_x - x)**2)
    else:
        pixelloss = torch.mean(torch.abs(recon_x-x))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return pixelloss + kl, pixelloss

def mseloss (recon_x,x):
    return torch.mean((recon_x - x)**2)

def main(args):
    # setup MRI data loaders
    
    train_ids, val_ids = create_train_idxs(395)

    if args.voxnorm:
        h5icov = "icov_x_z_anon.h5" if args.anon else "icov_x_z.h5"
        # if not os.path.exists("icov_x_z.h5"):
        #     print("calculating voxel stats")
        #     calc_icov(train_ids)
        #     print("done")
        with h5py.File(h5icov, "r") as hfstat:
            voxmean = torch.Tensor(np.moveaxis(hfstat["mean"][:],3,0))
            voxstd = torch.sqrt(torch.Tensor(np.moveaxis(hfstat["var"][:],3,0)))
        if args.cuda:
            voxmean = voxmean.cuda()
            voxstd = voxstd.cuda()
    if args.anon:
        mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask_defaced.nii").get_fdata()
    else:
        mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask.nii").get_fdata()
    if args.cuda:
        mask = torch.Tensor(mask).cuda()
    else:
        mask = torch.Tensor(mask)
    mask = mask[:192,2:226,:192]

    params = {'batch_size': args.batch,
          'shuffle': False,
          'num_workers': 0}
    # train_loader, test_loader
    h5file = "x_z_anon.h5" if args.anon else "x_z.h5"
    training_set = Dataset3dH5(train_ids, h5file=h5file)
    train_loader = data.DataLoader(training_set, **params)
    testing_set = Dataset3dH5(val_ids, h5file=h5file)
    test_loader = data.DataLoader(testing_set, **params)

    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # setup the VAE
    vae = VAE(z_dim=args.num_zdim, use_cuda=args.cuda, use_resblocks=args.resblocks, n_downsample=args.ndownsample, maxchannel=args.maxchannels, zchannel=16, variational=not args.mse)
    
    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
    #optimizer = optim.SGD(vae.parameters(), lr=args.learning_rate, weight_decay=0.01)

    train_elbo = []
    test_elbo = []

    startepoch = 0
    argshash = f"{args.num_zdim}_{args.maxchannels}_{args.ndownsample}{'_mse' if args.mse else '_vae'}{'_l1' if args.l1loss else ''}{'_rb' if args.resblocks else ''}{'voxnorm' if args.voxnorm else ''}{'_maskloss' if args.maskloss else ''}{'_addmin' if args.addmin else ''}"
    checkpointpath = f'checkpoint/vae_{argshash}.pt'
    if os.path.exists(checkpointpath):
        print(f"loading {checkpointpath}")
        state = torch.load(checkpointpath, map_location=torch.device(device))
        vae.load_state_dict(state['model'])
        for g in state["optimizer"]["param_groups"]:
            g['lr'] = args.learning_rate
        optimizer.load_state_dict(state["optimizer"])
        train_elbo = state["train_elbo"]
        test_elbo = state["test_elbo"]
        startepoch = len(train_elbo)
        del state['model']
        del state['optimizer']


    # training loop
    for epoch in range(startepoch,args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        count=0
        for (x, _) in tqdm(train_loader):
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
            
            if args.voxnorm:
                x = (x-voxmean)/voxstd*mask
            else:
                x = x*mask
                if args.addmin:
                    x = x-x.min()
            optimizer.zero_grad()
            if not args.mse:
                recon, mu, logvar = vae(x)
                if args.maskloss:
                    recon = recon*mask
                loss, mse = loss_function(recon, x, mu, logvar, mse=not args.l1loss)
            else:
                recon = vae(x)
                if args.maskloss:
                    recon = recon*mask
                loss = mseloss(recon, x)
                mse = loss

            # do ELBO gradient and accumulate loss
            loss.backward()
            optimizer.step()

            #total_norm = np.sum([p.grad.data.norm(2).item() for p in vae.parameters()])
            #print(f"gradient norm: {total_norm}")

            steploss = loss.item()
            count += 1           
            epoch_loss += steploss

            if args.verbose:
                print(f"loss for step: {steploss}, mse: {mse}, running mean: {epoch_loss/count}, previous epochs: {np.mean(train_elbo)}")

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train * args.batch
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
        
        checkpointpath = f'checkpoint/vae_{argshash}.pt'

        if epoch % args.test_frequency == 0 and len(test_loader)>0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            #for i, x in enumerate(test_loader):
            with torch.no_grad():
                for (x, _) in test_loader:
                    # if on GPU put mini-batch into CUDA memory
                    if args.cuda:
                        x = x.cuda()
                    if args.voxnorm:
                        x = (x-voxmean)/voxstd*mask
                    else:
                        x = x*mask
                        if args.addmin:
                            x = x-x.min()
                    # compute ELBO estimate and accumulate loss

                    if not args.mse:
                        mu, logvar = vae.encode(x)
                        recon =  vae.decode(mu)
                        if args.maskloss:
                            recon = recon*mask
                        loss, mse = loss_function(recon, x, mu, logvar, mse=not args.l1loss)
                    else:
                        recon = vae(x)
                        if args.maskloss:
                            recon = recon*mask
                        loss = mseloss(recon, x)
                    test_loss += loss.item()

            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test * args.batch
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

        torch.save({
                "args": args,
                "model": vae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_elbo": train_elbo,
                "test_elbo": test_elbo
            }, checkpointpath)
        if min(test_elbo) == test_elbo[-1]:
            torch.save({
                    "args": args,
                    "model": vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_elbo": train_elbo,
                    "test_elbo": test_elbo
                }, checkpointpath + '_best.pt')

    return vae


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=1, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-4, type=float, help='learning rate')
    parser.add_argument('-wd', '--weightdecay', default=0.001, type=float, help='weight decay')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('-nb', '--batch', default=1, type=int, help='batch size')
    parser.add_argument('-nz', '--num-zdim', default=512, type=int, help='latent dimensionality')
    parser.add_argument('-nc', '--maxchannels', default=128, type=int, help='latent dimensionality')
    parser.add_argument('-nd', '--ndownsample', default=3, type=int, help='number of downsampling steps (maximum is 5)')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('--mse', action='store_true', default=False, help='whether to not use variational loss')
    parser.add_argument('--l1loss', action='store_true', default=False, help='whether to use l1 loss')
    parser.add_argument('--resblocks', action='store_true', default=False, help='whether to use resblocks')
    parser.add_argument('--voxnorm', action='store_true', default=False, help='whether to subtract mean')
    parser.add_argument('--addmin', action='store_true', default=False, help='whether to subtract mean')
    parser.add_argument('--maskloss', action='store_true', default=False, help='whether to compute the loss only inside the mask')
    parser.add_argument('--anon', action='store_true', default=False, help='whether to use anonymized dataset')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='')

    args = parser.parse_args()
    print(str(args))

    model = main(args)