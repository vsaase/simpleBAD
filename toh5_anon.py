import argparse
import nibabel as nb
import numpy as np
import torch
from torch.utils import data
from torch import nn, optim
import torch.nn.functional as F
import random
from mri_dataloaders import Dataset3dNonlinearAnonymized
import pandas as pd
from tqdm import tqdm
import os
import h5py, hdf5plugin

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def ztransform(x2, mask):
    nmask = mask.sum()
    x = x2*mask
    mean = x.sum(dim=[2,3,4], keepdim=True)/nmask
    std = torch.sqrt((((x-mean*mask)**2)).sum(dim=[2,3,4], keepdim=True)/(nmask-1))
    xout = (x2-mean)/std
    return xout

def main(args):
    df = pd.read_csv("samples_anon_healthy.csv")
    ids = list(range(len(df)))
    
    params_train = {'batch_size': 1,
        'shuffle': False,
        'num_workers': 0,
        'drop_last': False}
    dset2 = Dataset3dNonlinearAnonymized(df.iloc[ids])
    loader2 = data.DataLoader(dset2, **params_train)
    mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask_defaced.nii").get_fdata()
    mask = torch.Tensor(mask)
    if args.cuda:
        mask = mask.cuda()
    j=-1
    with h5py.File(f"x_{args.normalizationmode}_anon.h5", "w") as hf:
        for (x2, _) in tqdm(loader2):
            if args.cuda:
                x2 = x2.cuda()
            if args.normalizationmode == "z":
                x2 = ztransform(x2, mask)
            elif args.normalizationmode == "logz":
                x2 = ztransform(torch.log(x2+1), mask)
            elif args.normalizationmode == "mean":
                x2 /= x2.mean(dim=[2,3,4], keepdim=True)
            elif args.normalizationmode == "logmean":
                x2 = torch.log(x2+1) - torch.log(x2.mean(dim=[2,3,4], keepdim=True))
            elif args.normalizationmode == "raw":
                x2 = x2
            else:
                exit(1)
            x2 = x2[:,:,:192,2:226,:192]
            
            j += 1
        
            if j == 0:
                print(f"creating dataset with size {x2.shape}")
                hf.create_dataset('x', data = x2.cpu().numpy().astype(np.float16), dtype=np.float16, chunks=(1,9,16,16,16), maxshape=(395,9, 192, 224, 192), **hdf5plugin.Blosc())
            else:
                hf["x"].resize((hf["x"].shape[0] + 1), axis = 0)
                hf["x"][-1:] = x2.cpu().numpy().astype(np.float16)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")

    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('-n','--normalizationmode', default="z", type=str, help='normalization mode')

    args = parser.parse_args()
    print(args)
    main(args)