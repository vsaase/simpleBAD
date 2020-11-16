import glob
import nibabel as nb
import numpy as np

niftis = glob.glob("/home/vsaase/data/nii/*/*_t2/*.nii.gz")

facemask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_face_mask.nii")
facemask = facemask.get_fdata() <= 0

for fnamenii in niftis:
    print(fnamenii)
    nii = nb.load(fnamenii)
    data = nii.get_fdata()
    if len(data.shape) == 4:
        data *= facemask[:,:,:,None]
    else:
        data *= facemask
    defaced = nb.Nifti1Image(data.astype(np.int16), header=nii.header, affine=nii.affine)
    defaced.header.set_data_dtype(np.int16)
    defaced.to_filename(fnamenii[:-7] + "_defaced.nii.gz")