import argparse
import h5py, hdf5plugin
import numpy as np
from copy import deepcopy
import torch
from util import create_train_idxs

def calc_icov(idxs="all", filename="x_z.h5", use_cuda=False):
    hf = h5py.File(filename, "r")
    x = hf["x"]
    chunks = x.chunks
    with h5py.File("icov_"+filename, "w") as hfout:
        for i in range(x.shape[2]//x.chunks[2]):
            for j in range(x.shape[3]//x.chunks[3]):
                for k in range(x.shape[4]//x.chunks[4]):
                    print([i,j,k])
                    maxi = (i+1)*chunks[2]
                    maxj = (j+1)*chunks[3]
                    maxk = (k+1)*chunks[4]
                    if idxs=="all":
                        r = x[:,:,
                            i*x.chunks[2]:maxi,
                            j*x.chunks[3]:maxj,
                            k*x.chunks[4]:maxk
                            ]
                    else:
                        r = np.concatenate([x[ii:ii+1,:,
                            i*x.chunks[2]:maxi,
                            j*x.chunks[3]:maxj,
                            k*x.chunks[4]:maxk
                            ] for ii in idxs], axis=0)
                        r = r.astype(np.float32)
                    r = torch.Tensor(r)
                    if use_cuda:
                        r = r.cuda()
                    mean = r.mean(dim=0, keepdim=True)
                    r -= mean
                    mean = mean.permute(2,3,4,1,0)[:,:,:,:,0]
                    cov = torch.matmul(r.permute(2,3,4,1,0), r.permute(2,3,4,0,1))/(r.shape[0]-1)
                    var = cov.diagonal(dim1=3, dim2=4)
                    #print(var.mean(dim=(0,1,2)))
                    #adding eps = mean variance / 100 for numerical stability
                    eps = torch.diag_embed(0*var + var.mean(dim=3, keepdim=True)/100)
                    icov = torch.inverse(cov + eps)
                    if i==0 and j==0 and k==0:
                        hfout.create_dataset('icov', data = icov.cpu().numpy().astype(np.float32), 
                            dtype=np.float32, chunks=(*(x.chunks[2:]),9,9) ,
                            maxshape=(192, 224, 192,9,9), **hdf5plugin.Blosc())
                        hfout.create_dataset('mean', data = mean.cpu().numpy().astype(np.float32), 
                            dtype=np.float32, chunks=(*(x.chunks[2:]),9) ,
                            maxshape=(192, 224, 192,9), **hdf5plugin.Blosc())
                        hfout.create_dataset('var', data = var.cpu().numpy().astype(np.float32), 
                            dtype=np.float32, chunks=(*(x.chunks[2:]),9) ,
                            maxshape=(192, 224, 192,9), **hdf5plugin.Blosc())
                    else:
                        s = hfout["icov"].shape
                        hfout["icov"].resize((max(maxi,s[0]),max(maxj,s[1]),max(maxk,s[2]),9,9))
                        hfout["icov"][i*x.chunks[2]:maxi,j*x.chunks[3]:maxj,k*x.chunks[4]:maxk] = icov.cpu().numpy().astype(np.float32)

                        hfout["mean"].resize((max(maxi,s[0]),max(maxj,s[1]),max(maxk,s[2]),9))
                        hfout["mean"][i*x.chunks[2]:maxi,j*x.chunks[3]:maxj,k*x.chunks[4]:maxk] = mean.cpu().numpy().astype(np.float32)

                        hfout["var"].resize((max(maxi,s[0]),max(maxj,s[1]),max(maxk,s[2]),9))
                        hfout["var"][i*x.chunks[2]:maxi,j*x.chunks[3]:maxj,k*x.chunks[4]:maxk] = var.cpu().numpy().astype(np.float32)

def mahalanobis2(x, filename = "icov_x_z.h5", use_cuda=False, modalityidx=None):
    out = torch.zeros((x.shape[0],1,*(x.shape[2:])), dtype=torch.float32)
    if modalityidx is None:
        modalityidx = np.arange(x.shape[1])
        compute_marginal = False
    else:
        compute_marginal = True
    with h5py.File(filename, "r") as hf:
        icov = hf["icov"]
        mean = hf["mean"]
        c = icov.chunks
        for i in range(x.shape[2]//c[0]):
            for j in range(x.shape[3]//c[1]):
                for k in range(x.shape[4]//c[2]):
                    #print([i,j,k])
                    maxi = (i+1)*c[0]
                    maxj = (j+1)*c[1]
                    maxk = (k+1)*c[2]
                    ic = icov[
                        i*c[0]:maxi,
                        j*c[1]:maxj,
                        k*c[2]:maxk
                        ]
                    if compute_marginal:
                        ic = torch.Tensor(ic)
                        cov = torch.inverse(ic)
                        cov = cov.numpy()
                        cov = cov[:,:,:,:,modalityidx]
                        cov = cov[:,:,:,modalityidx,:]
                        cov = torch.Tensor(cov)
                        ic = torch.inverse(cov)
                    else:
                        ic = ic[:,:,:,:,modalityidx]
                        ic = ic[:,:,:,modalityidx,:]
                        ic = torch.Tensor(ic)
                    mean0 = mean[i*c[0]:maxi,j*c[1]:maxj,k*c[2]:maxk]
                    mean0 = mean0[:,:,:,modalityidx]
                    mean0 = torch.Tensor(mean0).unsqueeze(4)
                    if use_cuda:
                        ic = ic.cuda()
                        out = out.cuda()
                        mean0 = mean0.cuda()
                    xin = x[0,:,i*c[0]:maxi,j*c[1]:maxj,k*c[2]:maxk].permute(1,2,3,0).unsqueeze(4)
                    xin -= mean0
                    out[0,0,i*c[0]:maxi,j*c[1]:maxj,k*c[2]:maxk] = torch.matmul(torch.matmul(ic, xin).permute(0,1,2,4,3),xin).squeeze()
        return out, xin.shape[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-i','--input', default="x_z.h5", type=str, help='input h5 file')
    args = parser.parse_args()
    print(args)

    train_ids, _ = create_train_idxs(395)
    calc_icov(train_ids, filename=args.input)