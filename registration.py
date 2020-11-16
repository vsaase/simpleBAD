import pandas as pd
import os, re
import hashlib
import nibabel as nb
import numpy as np
from functools import partial
import SimpleITK as sitk
import multiprocessing

from scipy.io import savemat, loadmat
from scipy.linalg import qr
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)

def nii2int16(path):
    nii = nb.load(path)
    data = np.round(nii.get_fdata()).astype(np.int16)                                               
    nii = nb.Nifti1Image(data, header=nii.header, affine=nii.affine)        
    nii.header.set_data_dtype(np.int16)                    
    nb.save(nii, path)

def antsAffineToOrthogonal(infilename, outfilename):
    m = loadmat(infilename)
    affine = np.reshape(m["AffineTransform_double_3_3"][:9,0], (3,3))
    Q,R = qr(affine)
    for i in range(3):
        if R[i,i] < 0:
            Q[:,i] *= -1
    m["AffineTransform_double_3_3"][:9,0] = np.reshape(Q,9)
    savemat(outfilename, m, format='4')

def biascorrect(infile, maskfile):
    inputImage = sitk.ReadImage(infile,sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    
    maskImage = sitk.ReadImage(maskfile, sitk.sitkUInt8)
    if len(inputImage.GetSize()) == 4:
        ext = sitk.ExtractImageFilter()
        size = list(inputImage.GetSize())
        nimg = size[3]
        size[3]=0
        subimgs = []
        for i in range(nimg):
            index = [0,0,0,i]
            ext.SetSize(size)
            ext.SetIndex(index)
            subImage = ext.Execute(inputImage)
            print(subImage.GetSize())
            subimgs += [corrector.Execute(subImage, maskImage)]
        output = sitk.JoinSeries(subimgs)
    else:
        output = corrector.Execute(inputImage, maskImage)

    sitk.WriteImage(output, infile)
    return infile

def run_MNI_nonlinear(workdir, seriesUIDs, strong=False):
    if not os.path.isdir(workdir):
        return False
    
    mask = f"mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_headmask.nii"
    names = ["T1","T1CE","FLAIR","T2","T2S","ADC","TRACEW"]

    uidhash = hashlib.md5("".join(seriesUIDs).encode("utf-8")).hexdigest()
    if strong: uidhash += "_strong"
    uidhash += "_n4"
    hashdir = f"{workdir}/{uidhash}"
    if not os.path.isdir(hashdir):
        os.makedirs(hashdir)
    print(f"registering {workdir}")
    mni = f"mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii"
    
    mprage = f"{workdir}/{seriesUIDs[7]}.nii.gz"
    mpragemni = f"{hashdir}/MPRAGEmni.nii.gz"

    if not os.path.isfile(mpragemni):
        if strong:
            cmd = (
                f"antsRegistration --dimensionality 3 --output [{hashdir}/MPRAGE_to_MNI,{mpragemni}] -v"
                f" --interpolation Linear --winsorize-image-intensities [0.005,0.995] --initial-moving-transform [{mni},{mprage},1] --use-histogram-matching"
                f" --transform Rigid[0.1] --metric MI[{mni},{mprage},1,32,Regular,0.25] --convergence [512x256x128,1e-6,10] --shrink-factors 4x2x1 --smoothing-sigmas 3x2x1vox"
                f" --transform Affine[0.1] --metric MI[{mni},{mprage},1,32,Regular,0.25] --convergence [512x256x128,1e-6,10] --shrink-factors 4x2x1 --smoothing-sigmas 3x2x1vox"
                f" --transform SyN[0.1,3,0] --metric CC[{mni},{mprage},1,4] --convergence [128x64x32x16,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox"
            )
        else:
            cmd = (
                f"antsRegistration --dimensionality 3 --output [{hashdir}/MPRAGE_to_MNI,{mpragemni}] -v"
                f" --interpolation Linear --winsorize-image-intensities [0.005,0.995] --initial-moving-transform [{mni},{mprage},1] --use-histogram-matching"
                f" --transform Rigid[0.1] --metric MI[{mni},{mprage},1,32,Regular,0.25] --convergence [512x256x128,1e-6,10] --shrink-factors 4x2x1 --smoothing-sigmas 3x2x1vox"
                f" --transform Affine[0.1] --metric MI[{mni},{mprage},1,32,Regular,0.25] --convergence [512x256x128,1e-6,10] --shrink-factors 4x2x1 --smoothing-sigmas 3x2x1vox"
                f" --transform SyN[0.1,3,0] --metric MeanSquares[{mni},{mprage},1,0] --convergence [100x70x50,1e-6,10] --shrink-factors 8x4x2 --smoothing-sigmas 3x2x1vox"
            )
        os.system(cmd)
        print(f"running N4 bias correction for {mpragemni}")
        biascorrect(mpragemni, maskfile=mask)
        nii2int16(mpragemni)
    
    #t1masked = f"{hashdir}/T1maskmni.nii.gz"
    #maskt1(t1,mni_mask,t1masked)

    for name, uid in zip(names, seriesUIDs):
        m = f"{workdir}/{uid}.nii.gz"
        if not os.path.isfile(m):
            return False
        
        if os.path.isfile(f"{workdir}/{uidhash}/{name}mni.nii.gz"):
            print(f"skipping {name}mni.nii.gz")
            continue
        else:
            if name == "TRACEW":
                #nii = nb.load(m)
                #niis = four_to_three(nii)
                cmd = (
                    f"antsApplyTransforms --interpolation Linear -v -d 3 -e 3"
                    f" -i {m} -r {mni} -o {hashdir}/{name}mni.nii.gz -t {hashdir}/MPRAGE_to_MNI1Warp.nii.gz -t {hashdir}/MPRAGE_to_MNI0GenericAffine.mat -t {hashdir}/ADC_to_MPRAGE0GenericAffine.mat"
                )
                #print(cmd)
                os.system(cmd)
            else:
                cmd = (
                    f"antsRegistration --dimensionality 3 --output {hashdir}/{name}_to_MPRAGE"
                    f" --interpolation Linear --winsorize-image-intensities [0.005,0.995] --initial-moving-transform [{mprage},{m},1]"
                    f" --transform Rigid[0.1] --metric MI[{mprage},{m},1,32,Regular,0.25] --convergence [512x256x128,1e-6,10] --shrink-factors 4x2x1 --smoothing-sigmas 3x2x1vox"
                )
                #print(cmd)
                os.system(cmd)
                cmd = (
                    f"antsApplyTransforms --interpolation Linear -v -d 3"
                    f" -i {m} -r {mni} -o {hashdir}/{name}mni.nii.gz -t {hashdir}/MPRAGE_to_MNI1Warp.nii.gz -t {hashdir}/MPRAGE_to_MNI0GenericAffine.mat -t {hashdir}/{name}_to_MPRAGE0GenericAffine.mat"
                )
                #print(cmd)
                os.system(cmd)
            print(f"running N4 bias correction for {hashdir}/{name}mni.nii.gz")
            biascorrect(f"{hashdir}/{name}mni.nii.gz", maskfile=mask)
            nii2int16(f"{hashdir}/{name}mni.nii.gz")
    
    return True


def run_MNI_T2(workdir, seriesUIDs, strong=False):
    if not os.path.isdir(workdir):
        return False
    
    names = ["T1","T1CE","FLAIR","T2","T2S","ADC","TRACEW","MPRAGE"]

    uidhash = hashlib.md5("".join(seriesUIDs).encode("utf-8")).hexdigest()
    uidhash += "_t2"
    hashdir = f"{workdir}/{uidhash}"
    if not os.path.isdir(hashdir):
        os.makedirs(hashdir)
    print(f"registering {workdir}")
    mni = f"mni_icbm152_nlin_asym_09c/mni_icbm152_t2_tal_nlin_asym_09c.nii"
    
    t2 = f"{workdir}/{seriesUIDs[3]}.nii.gz"
    t2mni = f"{hashdir}/T2mni.nii.gz"

    if not os.path.isfile(t2mni):
        cmd = (
            f"antsRegistration --dimensionality 3 --output [{hashdir}/T2_to_MNI,{t2mni}] -v"
            f" --interpolation Linear --winsorize-image-intensities [0.005,0.995] --initial-moving-transform [{mni},{t2},1] --use-histogram-matching"
            f" --transform Rigid[0.1] --metric MI[{mni},{t2},1,32,Regular,0.25] --convergence [512x256,1e-6,10] --shrink-factors 4x2 --smoothing-sigmas 2x1vox"
            f" --transform Affine[0.1] --metric MI[{mni},{t2},1,32,Regular,0.25] --convergence [512x256,1e-6,10] --shrink-factors 4x2 --smoothing-sigmas 2x1vox"
        )
        os.system(cmd)
        nii2int16(t2mni)

    for name, uid in zip(names, seriesUIDs):
        m = f"{workdir}/{uid}.nii.gz"
        if not os.path.isfile(m):
            return False
        
        if os.path.isfile(f"{workdir}/{uidhash}/{name}mni.nii.gz"):
            print(f"skipping {name}mni.nii.gz")
            continue
        else:
            if name == "TRACEW":
                #nii = nb.load(m)
                #niis = four_to_three(nii)
                cmd = (
                    f"antsApplyTransforms --interpolation Linear -v -d 3 -e 3"
                    f" -i {m} -r {mni} -o {hashdir}/{name}mni.nii.gz -t {hashdir}/T2_to_MNI0GenericAffine.mat -t {hashdir}/ADC_to_MPRAGE0GenericAffine.mat"
                )
                #print(cmd)
                os.system(cmd)
            else:
                cmd = (
                    f"antsRegistration --dimensionality 3 --output {hashdir}/{name}_to_MPRAGE -v"
                    f" --interpolation Linear --winsorize-image-intensities [0.005,0.995] --initial-moving-transform [{t2},{m},1] --use-histogram-matching"
                    f" --transform Rigid[0.1] --metric MI[{t2},{m},1,32,Regular,0.25] --convergence [512x256,1e-6,10] --shrink-factors 4x2 --smoothing-sigmas 2x1vox"
                )
                #print(cmd)
                os.system(cmd)
                cmd = (
                    f"antsApplyTransforms --interpolation Linear -v -d 3"
                    f" -i {m} -r {mni} -o {hashdir}/{name}mni.nii.gz -t {hashdir}/T2_to_MNI0GenericAffine.mat -t {hashdir}/{name}_to_MPRAGE0GenericAffine.mat"
                )
                #print(cmd)
                os.system(cmd)
            nii2int16(f"{hashdir}/{name}mni.nii.gz")
    
    return True

def inverse_MNI_T2(infile, outfile, workdir, seriesUIDs):
    uidhash = hashlib.md5("".join(seriesUIDs).encode("utf-8")).hexdigest()
    uidhash += "_t2"
    hashdir = f"{workdir}/{uidhash}"
    mni = f"mni_icbm152_nlin_asym_09c/mni_icbm152_t2_tal_nlin_asym_09c.nii"
    
    t2 = f"{workdir}/{seriesUIDs[3]}.nii.gz"
    if not os.path.isfile(f"{hashdir}/T2_to_MNIorth0GenericAffine.mat"):
        cmd = (
            f"antsRegistration --dimensionality 3 --output {hashdir}/T2_to_MNIorth -v"
            f" --interpolation Linear --winsorize-image-intensities [0.005,0.995] --initial-moving-transform [{mni},{t2},1] --use-histogram-matching"
            f" --transform Rigid[0.1] --metric MI[{mni},{t2},1,32,Regular,0.25] --convergence [512x256,1e-6,10] --shrink-factors 4x2 --smoothing-sigmas 2x1vox"
        )
        os.system(cmd)
    cmd = (
        f"antsApplyTransforms --interpolation Linear -v -d 3 -e 3"
        f" -i {infile} -r {mni} -o {outfile} -t {hashdir}/T2_to_MNIorth0GenericAffine.mat -t [{hashdir}/T2_to_MNI0GenericAffine.mat,1] "
    )
    #print(cmd)
    os.system(cmd)


def reg(i, df, basedir, strong=False, method="T2"):
    print(i)
    studyUID = df.iloc[i,1]        
    seriesUIDs = df.iloc[i,2:10]
    uidhash = hashlib.md5("".join(seriesUIDs).encode("utf-8")).hexdigest()
    if method != "T2":
        if strong: uidhash += "_strong"
        uidhash += "_n4"
    directory = f"{basedir}/nii/{studyUID}"
    hashdir = f"{basedir}/nii/{studyUID}/{uidhash}"
    print(hashdir)
    for j, name in enumerate(["T1","T1CE","FLAIR","T2","T2S","ADC","TRACEW","MPRAGE"]):
        if not os.path.isfile(f"{hashdir}/{name}mni.nii.gz"):
            if method=="T2":
                run_MNI_T2(directory, seriesUIDs, strong=strong)
            else:
                run_MNI_nonlinear(directory, seriesUIDs, strong=strong)
            break
        else:
            print(f"found file {hashdir}/{name}mni.nii.gz")


def allRegistrations(basedir, nproc=1, strong=False, method="T2"):
    df = pd.read_csv("patho_sample44_t1_t1ce_flairfs_t2_t2star_adc_tracew_mprage.csv")
    if nproc == 1:
        for i in range(len(df)):
            reg(i, df, basedir, strong=strong)
    else:
        pool = multiprocessing.Pool(nproc)
        pool.map(partial(reg, df=df, basedir=basedir, strong=strong, method=method), range(len(df)))
