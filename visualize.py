import h5py, hdf5plugin
import numpy as np
import nibabel as nb
import pandas as pd
from mri_dataloaders import Dataset3dNonlinear
from torch.utils import data
from tensor2nifti import tensor2nifti_mprage
import os
import torch
import matplotlib.pyplot as plt

mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask.nii").get_fdata()[:192,2:226,:192][None,None,:,:,:]
hfp = h5py.File("test_patho/r.h5","r")
rp = hfp["r"]

replicationpad = torch.nn.ReplicationPad3d((0,1,2,3,0,1))

for j in range(12,44):
    dfpatho = pd.read_csv("patho_sample44_t1_t1ce_flairfs_t2_t2star_adc_tracew_mprage.csv")[j:j+1]
    params = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 0}
    dset1 = Dataset3dNonlinear(dfpatho, suffix="_t2")
    test_loader = data.DataLoader(dset1, **params)

    (x, info) = iter(test_loader).next()
    pathx = tensor2nifti_mprage(x.detach().cpu(), f'temp_x{j}')

        
    paths = []
    for i in range(4):
        x0 = torch.Tensor(rp[j:j+1,i:i+1]*mask)
        x0 = replicationpad(x0)
        path = tensor2nifti_mprage(x0, f'temp{i}')
        paths.append(path)
    os.system(f"itksnap -g {pathx} -o {' '.join(paths[:2])} {pathx} {' '.join(paths[2:])}")


s = torch.load("checkpoint/vae_512_128_3_mse_rbvoxnorm.pt",map_location=torch.device("cpu")
plt.plot(s["train_elbo"], label="train")
plt.plot(s["test_elbo"], label="test")
plt.title("convergence diagnostics")
plt.xlabel("epoch")
plt.ylabel("L2 loss")
plt.legend()
plt.show()