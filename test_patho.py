import torch
import argparse
from torch.utils import data
from mri_dataloaders import Dataset3dNonlinear, Dataset3dNonlinearAnonymized
from util import create_train_idxs
from vae import VAE
from inversecovariance import mahalanobis2
from tqdm import tqdm
import nibabel as nb
import random
import pandas as pd
from toh5 import ztransform
from copy import deepcopy
import numpy as np
import h5py, hdf5plugin
from pcakernel import pca_project_test, direct_project

@torch.no_grad()
def main(args):
    _, val_ids = create_train_idxs(395)

    with h5py.File(f"icov_x_z.h5", "r") as hfstat:
        voxmean = torch.Tensor(np.moveaxis(hfstat["mean"][:],3,0))
        voxstd = torch.sqrt(torch.Tensor(np.moveaxis(hfstat["var"][:],3,0)))

    if args.anon:
        mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask_defaced.nii").get_fdata()
    else:
        mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask.nii").get_fdata()
    mask = torch.Tensor(mask)
    maskc = mask[:192,2:226,:192]
    
    params = {'batch_size': 1,
        'shuffle': False,
        'num_workers': 0}
    if args.anon:
        dfpatho = pd.read_csv("samples_anon_patho.csv")
        dset1 = Dataset3dNonlinearAnonymized(dfpatho)
    else:
        dfpatho = pd.read_csv("patho_sample44_t1_t1ce_flairfs_t2_t2star_adc_tracew_mprage.csv")
        dset1 = Dataset3dNonlinear(dfpatho, suffix="_t2")
    test_loader = data.DataLoader(dset1, **params)

    vae = VAE(z_dim=512, use_cuda=True, use_resblocks=True, n_downsample=3, maxchannel=128, zchannel=16, variational=False)
    checkpointpath = f'checkpoint/vae_512_128_3_mse_rbvoxnorm.pt_best.pt'
    s = torch.load(checkpointpath, map_location=torch.device("cuda"))
    vae.load_state_dict(s["model"])
    del s
    j=0
    with h5py.File(f"test_patho/r.h5", "w") as hf:
        for (x, _) in tqdm(test_loader):
            x = ztransform(x, mask)
            x = x[:,:,:192,2:226,:192]
            xnorm = (x-voxmean)/voxstd*maskc

            rsimple = (xnorm**2).sum(dim=1, keepdim=True)
            h5icov = "icov_x_z_anon.h5" if args.anon else "icov_x_z.h5"
            rmvn, _ = mahalanobis2(deepcopy(x), filename=h5icov, use_cuda=False)

            _, rpca = direct_project(x, "z", anon=args.anon)
            rpca = (rpca**2).sum(dim=1, keepdim=True)

            recon = vae(xnorm.cuda()).cpu()
            rvae = ((xnorm-recon)**2).sum(dim=1, keepdim=True)

            r = torch.cat([
                rsimple, 
                rmvn, 
                rpca, 
                rvae,
            ], dim=1)
            
            if j == 0:
                print(f"creating dataset with size {r.shape}")
                hf.create_dataset('r', data = r.cpu().numpy().astype(np.float32), dtype=np.float32, chunks=(1,r.shape[1],16,16,16), maxshape=(len(val_ids), r.shape[1], 192, 224, 192), **hdf5plugin.Blosc())
            else:
                hf["r"].resize((hf["r"].shape[0] + 1), axis = 0)
                hf["r"][-1:] = r.cpu().numpy().astype(np.float32)
            j += 1
            del r, rpca
            del rvae
            del rsimple, rmvn,xnorm,x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--anon', action='store_true', default=False, help='whether to use anonymized dataset')
    args = parser.parse_args()
    main(args)