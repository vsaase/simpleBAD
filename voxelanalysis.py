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
    mask = mask[:192,2:226,:192]

    hf = h5py.File("test_healthy/r.h5","r")
    rh = hf["r"]

    hfp = h5py.File("test_patho/r.h5", "r")
    rp = hfp["r"]
    if args.anon:
        dfpatho = pd.read_csv("samples_anon_patho.csv")
    else:
        dfpatho = pd.read_csv("patho_sample44_t1_t1ce_flairfs_t2_t2star_adc_tracew_mprage.csv")

    # voxelwise
    nbatch=44
    ap=np.zeros((4,nbatch))
    auc=np.zeros((4,nbatch))
    for j in range(4):
        for i in range(nbatch):
            labels = np.array([])
            scores = np.array([])
            if args.anon:
                studyid = dfpatho.loc[i]["Study Instance UID"]
            else:
                studyid = dfpatho.loc[i]["Study Instance UID"]
                seg = nb.load(f"segment_patho/{studyid}_seg.nii.gz").get_fdata()[:192,2:226,:192] == 1.0
            img = rp[i,j]
            labels = np.concatenate((labels, seg[mask>0]))
            scores = np.concatenate((scores, img[mask>0]))
            imgh = rh[i,j]
            labels = np.concatenate((labels, 0*seg[mask>0]))
            scores = np.concatenate((scores, imgh[mask>0]))
                
            ap[j,i] = average_precision_score(labels, scores)
            auc[j,i] = roc_auc_score(labels, scores)
            print(i)

    np.savez("apauc", [ap, auc])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--anon', action='store_true', default=False, help='whether to use anonymized dataset')
    args = parser.parse_args()
    main(args)