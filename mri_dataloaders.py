import torch
from torch.utils import data
import numpy as np
import nibabel as nb
import hashlib
import sys
import os
from io import BytesIO
from copy import deepcopy
import h5py, hdf5plugin
import re
import getpass


class Dataset3dNonlinearAnonymized(data.Dataset):
    basedir = f"./data_anonymized"
    basedirin = basedir

    def __init__(self, df, seriesnames = ["T1","T1 CE","FLAIR","T2","T2S","ADC","TRACEW","MPRAGE"]):
        'Initialization'
        self.df = df # mask_existing_zipmprage(df, seriesnames = seriesnames) #mask_existing_mni(df, seriesnames = seriesnames) #
        self.seriesnames = seriesnames

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.df.iloc[index]
        studyhash = row["studyhash"]
        processinghash = row["processinghash"]

        nchan = len(self.seriesnames)
        if "TRACEW" in self.seriesnames:
            nchan += 1
            
        s0 = np.array([193,229,193])
        x = np.zeros(np.append([nchan],s0), dtype=np.int16)
        idx = 0
        for _i, name in enumerate(self.seriesnames):
            if name=="T1 CE": name = "T1CE"
            fname = f"{self.basedirin}/nii/{studyhash}/{processinghash}/{name}mni.nii.gz"
            try:
                nii = nb.load(fname)
                data = nii.get_fdata()
            except Exception as e:
                print(f"error processing {studyhash}/{processinghash}/{name}")
                raise(e)
            if name == "TRACEW":
                x[[idx,idx+1],:,:,:] = np.moveaxis(data[:,:,:,:2],-1,0)
                idx += 2
            else:
                x[idx,:,:,:] = data
                idx += 1

        return x.astype(np.float32), {"basedir": self.basedir, "basedirin": self.basedirin, "studyhash": studyhash, "processingshash": processinghash}

class Dataset3dNonlinear(data.Dataset):
    basedir = f"/home/{getpass.getuser()}/data"
    basedirin = basedir

    def __init__(self, df, seriesnames = ["T1","T1 CE","FLAIR","T2","T2S","ADC","TRACEW","MPRAGE"], suffix=""):
        'Initialization'
        self.df = df # mask_existing_zipmprage(df, seriesnames = seriesnames) #mask_existing_mni(df, seriesnames = seriesnames) #
        self.seriesnames = seriesnames
        self.suffix = suffix

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.df.iloc[index]
        studyUID = row["Study Instance UID"]    
        seriesUIDs = [row[f"Series Instance UID {name}"] for name in self.seriesnames]
        uidhash = hashlib.md5("".join(seriesUIDs).encode("utf-8")).hexdigest()

        nchan = len(self.seriesnames)
        if "TRACEW" in self.seriesnames:
            nchan += 1
        # Load data and get label
        # try:
        #     with h5py.File(f"{basedir}/h5mni9/{studyUID}/{uidhash}/mni.h5", 'r') as f:
        #         x = f["data"][:]
        # except Exception as e:
        #     print(e)
        s0 = np.array([193,229,193])
        x = np.zeros(np.append([nchan],s0), dtype=np.int16)
        idx = 0
        for i, name in enumerate(self.seriesnames):
            if name=="T1 CE": name = "T1CE"
            fname = f"{self.basedirin}/nii/{studyUID}/{uidhash}{self.suffix}/{name}mni.nii.gz"
            try:
                nii = nb.load(fname)
                data = nii.get_fdata()
            except Exception as e:
                print(f"error processing {studyUID}/{uidhash}{self.suffix}/{name}")
                raise(e)
            if name == "TRACEW":
                x[[idx,idx+1],:,:,:] = np.moveaxis(data[:,:,:,:2],-1,0)
                idx += 2
            else:
                x[idx,:,:,:] = data
                idx += 1

        return x.astype(np.float32), {"basedir": self.basedir, "basedirin": self.basedirin, "studyUID": studyUID, "uidhash": uidhash, "mprageid":seriesUIDs[-1], "seriesUIDs":seriesUIDs}

class Dataset3dH5(data.Dataset):
    def __init__(self, idxs, h5file = "x_z.h5"):
        'Initialization'
        self.idxs = idxs
        hf = h5py.File(h5file, "r")
        self.x = hf['x']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.idxs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        idx = self.idxs[index]
        return torch.Tensor(self.x[idx].astype(np.float32)), 0
