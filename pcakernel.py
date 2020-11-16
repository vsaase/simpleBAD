import argparse
import nibabel as nb
import numpy as np
import os
import time
import torch
from copy import deepcopy
import h5py, hdf5plugin
from util import create_train_idxs


def direct_project(y, normalizationmode, nostd=False, anon=False):
    train_ids, _ = create_train_idxs(395)
    if anon:
        mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask_defaced.nii").get_fdata()
    else:
        mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask.nii").get_fdata()
    mask = mask[:192,2:226,:192][None,None,:,:,:]
    mask = torch.Tensor(mask)
    
    with h5py.File(f"icov_x_{normalizationmode}.h5", "r") as hfout:
        mean = np.moveaxis(hfout["mean"][:],3,0)[None,:,:,:,:].astype(np.float32)
        std = np.sqrt(np.moveaxis(hfout["var"][:],3,0))[None,:,:,:,:].astype(np.float32)
    y_z = (y-mean)/(1.0 if nostd else std)
    residual = deepcopy(y_z)
    with h5py.File(f"x_{normalizationmode}.h5", "r") as hf:
        x = hf['x']
        for ii,i in enumerate(train_ids):
            print(f"{ii},{i}")
            xi_z = torch.Tensor((x[i].astype(np.float32)-mean)/(1.0 if nostd else std))
            xi_norm = torch.norm(xi_z*mask)
            coef = (residual*xi_z*mask).sum()/xi_norm
            print(coef)
            residual = residual - coef * xi_z/xi_norm
            #print(np.linalg.norm(residual))
    return y_z - residual, residual

def pca_project(y, normalizationmode, nostd=False):
    train_ids, _ = create_train_idxs(395)
    #train_ids = train_ids[:10]
    #kernelrow = compute_kernelrow(y, normalizationmode, nostd)
    kernelrow = compute_kernelrow_local(y, normalizationmode, nostd)
    print(kernelrow)
    with h5py.File(f"localkernel_{normalizationmode}{'_nostd' if nostd else ''}.h5", "r") as hfout:
        kernel = torch.Tensor(hfout['kernel'][0,0,0])
    kernel = kernel[train_ids,:][:,train_ids]
    eps = kernel.diagonal().mean()/10
    coefrow = torch.matmul(kernelrow[0,0,0], torch.inverse(kernel + eps*torch.eye(kernel.shape[-1])))
    projection = 0*y
    with h5py.File(f"icov_x_{normalizationmode}.h5", "r") as hfout:
        mean = np.moveaxis(hfout["mean"][:],3,0)[None,:,:,:,:]
        std = np.sqrt(np.moveaxis(hfout["var"][:],3,0))[None,:,:,:,:]
    y_z = (y-mean)/(1.0 if nostd else std)
    with h5py.File(f"x_{normalizationmode}.h5", "r") as hf:
        x = hf['x']
        for ii,i in enumerate(train_ids):
            print(coefrow[0,ii])
            print(ii)
            projection += coefrow[0,ii] * (x[i].astype(np.float32)-mean)/(1.0 if nostd else std)
    return projection, y_z - projection

def pca_project_test(test_ids, cuda=False):
    train_ids, _ = create_train_idxs(395)
    with h5py.File(f"localkernel_z.h5", "r") as hfout:
        kernel = torch.Tensor(hfout['kernel'][0,0,0])
        if cuda: kernel.cuda()
    kernelrows = kernel[test_ids,:][:,train_ids]
    kernel = kernel[train_ids,:][:,train_ids]
    eps = kernel.diagonal().mean()/10
    epseye = torch.eye(kernel.shape[-1])
    if cuda: epseye.cuda()
    coefrows = torch.matmul(kernelrows, torch.inverse(kernel + eps*epseye))
    print(coefrows)
    residuals = torch.zeros(len(test_ids),9,192,224,192)
    if cuda: residuals.cuda()
    with h5py.File(f"icov_x_z.h5", "r") as hfout:
        mean = torch.Tensor(np.moveaxis(hfout["mean"][:],3,0)[None,:,:,:,:])
        std = torch.Tensor(np.sqrt(np.moveaxis(hfout["var"][:],3,0))[None,:,:,:,:])
        if cuda:
            mean.cuda()
            std.cuda()
    print("loading h5 file")
    with h5py.File(f"x_z.h5", "r") as hf:
        x = hf['x']
        print("loading test data")
        for ii,i in enumerate(test_ids):
            print(ii)
            xi = torch.Tensor(x[i].astype(np.float32))
            if cuda: xi.cuda()
            residuals[ii] = (xi-mean)/std
        print("updating residuals")
        for ii,i in enumerate(train_ids):
            xi = torch.Tensor(x[i].astype(np.float32))
            if cuda: xi.cuda()
            print(ii)
            residuals -= coefrows[:,ii].view(-1,1,1,1,1) * (xi-mean)/std
            #print((residuals**2).sum())
    return residuals

def compute_kernelrow_local(y, normalizationmode, nostd=False, anon=False):
    if anon:
        mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask_defaced.nii").get_fdata()
    else:
        mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask.nii").get_fdata()
    mask = mask[:192,2:226,:192][None,None,:,:,:]
    nmask = mask.sum()
    mask = torch.Tensor(mask)
    train_ids, _ = create_train_idxs(395)
    #train_ids = train_ids[:10]
    with h5py.File(f"icov_x_{normalizationmode}.h5", "r") as hfout:
        mean = np.moveaxis(hfout["mean"][:],3,0)[None,:,:,:,:]
        std = np.sqrt(np.moveaxis(hfout["var"][:],3,0))[None,:,:,:,:]
    y_z = (y-mean)/(1.0 if nostd else std)
    with h5py.File(f"x_{normalizationmode}.h5", "r") as hf:
        x = hf['x']
        n = len(train_ids)
        chunks = x.chunks
        kmeanrow = torch.zeros(1,1,1,1,n)
        Nmask = mask.sum()
        for i in range(x.shape[2]//x.chunks[2]):
            for j in range(x.shape[3]//x.chunks[3]):
                for k in range(x.shape[4]//x.chunks[4]):
                    print([i,j,k])
                    maxi = (i+1)*chunks[2]
                    maxj = (j+1)*chunks[3]
                    maxk = (k+1)*chunks[4]

                    nmask = mask[:,:,
                        i*x.chunks[2]:maxi,
                        j*x.chunks[3]:maxj,
                        k*x.chunks[4]:maxk
                        ].sum()
                    if nmask == 0:
                        continue
                    r = np.concatenate([x[ii:ii+1,:,
                        i*x.chunks[2]:maxi,
                        j*x.chunks[3]:maxj,
                        k*x.chunks[4]:maxk
                        ] for ii in train_ids], axis=0)
                    r = r.astype(np.float32)
                    ry = y_z[:,:,
                            i*x.chunks[2]:maxi,
                            j*x.chunks[3]:maxj,
                            k*x.chunks[4]:maxk
                        ]
                    r = torch.Tensor(r)
                    #mean = r.mean(dim=0, keepdim=True)
                    r -= mean[:,:,
                            i*x.chunks[2]:maxi,
                            j*x.chunks[3]:maxj,
                            k*x.chunks[4]:maxk
                        ]
                    #ry -= mean
                    #std = r.std(dim=0, keepdim=True)
                    if not nostd:
                        r /= std[:,:,
                                i*x.chunks[2]:maxi,
                                j*x.chunks[3]:maxj,
                                k*x.chunks[4]:maxk
                            ]
                    #ry /= std
                    r = r*mask[:,:,
                        i*x.chunks[2]:maxi,
                        j*x.chunks[3]:maxj,
                        k*x.chunks[4]:maxk
                        ]
                    ry = ry*mask[:,:,
                        i*x.chunks[2]:maxi,
                        j*x.chunks[3]:maxj,
                        k*x.chunks[4]:maxk
                        ]
                    r = r.view(r.shape[0],-1)
                    ry = ry.view(ry.shape[0],-1)
                
                    localkernelrow = torch.matmul(ry, torch.transpose(r,0,1)).view(1,1,1,1,n)/nmask
                    kmeanrow += localkernelrow*nmask/Nmask
                    print(kmeanrow[0,0,0,0,0])
    return kmeanrow

def compute_kernelrow(y, normalizationmode, nostd=False):
    mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask.nii").get_fdata()
    mask = mask[:192,2:226,:192][None,None,:,:,:]
    mask = torch.Tensor(mask)
    Nmask = mask.sum()
    train_ids, _ = create_train_idxs(395)
    with h5py.File(f"icov_x_{normalizationmode}.h5", "r") as hfout:
        mean = np.moveaxis(hfout["mean"][:],3,0)[None,:,:,:,:]
        std = np.sqrt(np.moveaxis(hfout["var"][:],3,0))[None,:,:,:,:]
    yz = (y-mean)
    if not nostd:
        yz /= std
    with h5py.File(f"x_{normalizationmode}.h5", "r") as hf:
        x = hf['x']
        kmeanrow = torch.zeros(1,1,1,1,len(train_ids))
        for i in train_ids:
            print(i)
            xi = torch.Tensor((x[i].astype(np.float32)-mean)/(1.0 if nostd else std))
            kmeanrow[0,0,0,0,i] = torch.sum((yz*xi)*mask/Nmask)
            print(kmeanrow[0,0,0,0,i])
    return kmeanrow
            

def main2(args):
    mask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask.nii").get_fdata()
    mask = mask[:192,2:226,:192][None,None,:,:,:]
    Nmask = mask.sum()
    mask = torch.Tensor(mask)
    with h5py.File(f"icov_x_{args.normalizationmode}.h5", "r") as hfout:
        mean = np.moveaxis(hfout["mean"][:],3,0)[None,:,:,:,:]
        std = np.sqrt(np.moveaxis(hfout["var"][:],3,0))[None,:,:,:,:]
    with h5py.File(f"x_{args.normalizationmode}.h5", "r") as hf:
        with h5py.File(f"localkernel_{args.normalizationmode}{'_nostd' if args.nostd else ''}.h5", "w") as hfout:
            x = hf['x']
            n = x.shape[0]
            chunks = x.chunks
            kmean = torch.zeros(1,1,1,n,n)
            for i in range(x.shape[2]//x.chunks[2]):
                for j in range(x.shape[3]//x.chunks[3]):
                    for k in range(x.shape[4]//x.chunks[4]):
                        print([i,j,k])
                        maxi = (i+1)*chunks[2]
                        maxj = (j+1)*chunks[3]
                        maxk = (k+1)*chunks[4]

                        nmask = mask[:,:,
                            i*x.chunks[2]:maxi,
                            j*x.chunks[3]:maxj,
                            k*x.chunks[4]:maxk
                            ].sum()
                        if nmask > 0:
                            r = x[:,:,
                                i*x.chunks[2]:maxi,
                                j*x.chunks[3]:maxj,
                                k*x.chunks[4]:maxk
                                ].astype(np.float32)
                            r = torch.Tensor(r)
                            r -= mean[:,:,
                                    i*x.chunks[2]:maxi,
                                    j*x.chunks[3]:maxj,
                                    k*x.chunks[4]:maxk
                                ]
                            if not args.nostd:
                                r /= std[:,:,
                                        i*x.chunks[2]:maxi,
                                        j*x.chunks[3]:maxj,
                                        k*x.chunks[4]:maxk
                                    ]
                            #mean = r.mean(dim=0, keepdim=True)
                            #r -= mean
                            #print(mean[0,:,0,0,0])
                            #std = r.std(dim=0, keepdim=True)
                            #r /= std
                            r = r*mask[:,:,
                                i*x.chunks[2]:maxi,
                                j*x.chunks[3]:maxj,
                                k*x.chunks[4]:maxk
                                ]
                            r = r.view(r.shape[0],-1)
                            localkernel = torch.matmul(r, torch.transpose(r,0,1)).view(1,1,1,n,n)/nmask
                            kmean += localkernel*nmask/Nmask
                            #print(kmean)
                        else:
                            localkernel = torch.zeros(1,1,1,n,n)
                        if i==0 and j==0 and k==0:
                            hfout.create_dataset('localkernel', data = localkernel.cpu().numpy().astype(np.float32), 
                                dtype=np.float32, chunks=(1,1,1,n,n) ,
                                maxshape=(*(x.chunks[2:]), n, n), **hdf5plugin.Blosc())
                        else:
                            s = hfout["localkernel"].shape
                            hfout["localkernel"].resize((max(i+1,s[0]),max(j+1,s[1]),max(k+1,s[2]),n,n))
                            hfout["localkernel"][i:i+1,j:j+1,k:k+1] = localkernel.cpu().numpy().astype(np.float32)
            hfout.create_dataset('kernel', data=kmean.cpu().numpy().astype(np.float32), dtype=np.float32, **hdf5plugin.Blosc())
                            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")

    parser.add_argument('-n','--normalizationmode', default="z", type=str, help='normalization mode')

    parser.add_argument('--nostd', action='store_true', default=False, help='whether not to divide by std')
    
    args = parser.parse_args()
    print(args)
    main2(args)