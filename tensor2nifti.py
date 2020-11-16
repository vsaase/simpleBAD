import nibabel as nb
import numpy as np
import os
from torch.nn import functional as F

def tensor2nifti_mprage(t, savepath):
    mni = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii")
    affine = mni.affine
    for i in range(t.shape[0]):
        data = t[i].numpy()
        data = np.moveaxis(data,0,-1)
        #import IPython; IPython.embed()
        nii = nb.Nifti1Image(data, affine)
        hdr = nii.header
        hdr.set_data_dtype(data.dtype)
        if i==0:
            path = savepath + ".nii.gz"
        else:
            path = savepath + "_" + str(i) + ".nii.gz"
        nii.to_filename(path)
    return path

def numpy2nifti_mprage(t, savepath):
    mni = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii")
    affine = mni.affine
    for i in range(t.shape[0]):
        data = t[i,:,:,:,:]
        data = np.moveaxis(data,0,-1)
        #import IPython; IPython.embed()
        nii = nb.Nifti1Image(data, affine)
        hdr = nii.header
        hdr.set_data_dtype(data.dtype)
        if i==0:
            path = savepath + ".nii.gz"
        else:
            path = savepath + "_" + str(i) + ".nii.gz"
        nii.to_filename(path)
    return path
