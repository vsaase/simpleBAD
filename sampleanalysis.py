import argparse
import h5py, hdf5plugin
import numpy as np
import nibabel as nb
import pandas as pd
from sklearn.metrics import *
from auc_delong_xu import auc_ci_Delong
import random

def main(args):
    if args.anon:
        mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask_defaced.nii").get_fdata()
    else:
        mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask.nii").get_fdata()
    mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask.nii").get_fdata()[:192,2:226,:192]

    hf = h5py.File("test_healthy/r.h5","r")
    rh = hf["r"]

    hfp = h5py.File("test_patho/r.h5", "r")
    rp = hfp["r"]

    # samplewise bootstrap
    scores_patho = np.zeros((4,44))
    scores_healthy = np.zeros((4,44))
    for j in range(4):
        for i in range(44):
            scores_patho[j,i] = rp[i,j][mask>0].mean()
            scores_healthy[j,i] = rh[i,j][mask>0].mean()

    nboot = 100000
    nsamples = 44
    ap=np.zeros((4,nboot))
    auc=np.zeros((4,nboot))
    for k in range(nboot):
        samples_patho = random.choices(range(44), k=nsamples)
        samples_healthy = random.choices(range(44), k=nsamples)
        for j in range(4):
            labels = np.array([])
            scores = np.array([])
            for i in range(nsamples):
                labels = np.concatenate((labels, [1]))
                scores = np.concatenate((scores, [scores_patho[j,samples_patho[i]]]))
                labels = np.concatenate((labels, [0]))
                scores = np.concatenate((scores, [scores_healthy[j,samples_healthy[i]]]))
                #print(average_precision_score(labels, scores))
            
            ap[j,k] = average_precision_score(labels, scores)
            auc[j,k] = roc_auc_score(labels, scores)
        print(f"{k} ap {ap[:,k]} auc {auc[:,k]}")


    # ap:  [0.70122438, 0.72801965, 0.72524272, 0.74725537]
    # auc: [0.76225747, 0.77321757, 0.78529345, 0.80142893]

    np.savez("apauc_sample", [ap, auc])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--anon', action='store_true', default=False, help='whether to use anonymized dataset')
    args = parser.parse_args()
    main(args)