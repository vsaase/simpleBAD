import torch
import argparse
from torch.utils import data
from mri_dataloaders import Dataset3dH5
from util import create_train_idxs
from vae import VAE
from inversecovariance import mahalanobis2
from tqdm import tqdm
import nibabel as nb
import numpy as np
from copy import deepcopy
import h5py, hdf5plugin
from pcakernel import pca_project_test, direct_project, pca_project

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
    mask = mask[:192,2:226,:192]
    params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 0}
    testing_set = Dataset3dH5(val_ids)
    test_loader = data.DataLoader(testing_set, **params)

    # due to limited memory size compute pca residuals in batches
    #batchsize=8
    #pcaresiduals = pca_project_test(val_ids[:batchsize])
    #offset=0

    vae = VAE(z_dim=512, use_cuda=True, use_resblocks=True, n_downsample=3, maxchannel=128, zchannel=16, variational=False)
    checkpointpath = f'checkpoint/vae_512_128_3_mse_rbvoxnorm.pt_best.pt'
    s = torch.load(checkpointpath, map_location=torch.device("cuda"))
    vae.load_state_dict(s["model"])
    del s
    j=0
    with h5py.File(f"test_healthy/r.h5", "w") as hf:
        for (x, _) in tqdm(test_loader):
            x = x
            xnorm = (x-voxmean)/voxstd*mask

            rsimple = (xnorm**2).sum(dim=1, keepdim=True)
            h5icov = "icov_x_z_anon.h5" if args.anon else "icov_x_z.h5"
            rmvn, _ = mahalanobis2(deepcopy(x), filename=h5icov, use_cuda=False)

            # if j-offset+1 > pcaresiduals.shape[0]:
            #     # next pca batch
            #     del pcaresiduals
            #     offset = j
            #     pcaresiduals = pca_project_test(val_ids[offset:min(offset+batchsize, len(val_ids))], cuda=False)
            # rpca = (pcaresiduals[j-offset:j-offset+1]**2).sum(dim=1, keepdim=True)

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
                hf.create_dataset('r', data = r.cpu().numpy().astype(np.float32), dtype=np.float32, chunks=(1,r.shape[1],16,16,16), maxshape=(len(val_ids), 4, 192, 224, 192), **hdf5plugin.Blosc())
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