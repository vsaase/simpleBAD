
import nibabel as nb
import numpy as np
import json
from datetime import datetime
import hashlib
import pandas as pd
import os

facemask = nb.load("mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_face_mask.nii")
facemask = facemask.get_fdata() <= 0

anonlist = [
    'PatientName',
    'InstitutionName',
    'InstitutionalDepartmentName',
    'InstitutionAddress',
    'StationName',
    #'SeriesInstanceUID',
    #'StudyInstanceUID',
    'ReferringPhysicianName',
    'StudyID',
    'PatientID',
    'AccessionNumber',
    'PatientBirthDate',
    'ProcedureStepDescription',
    'AcquisitionTime',
    'AcquisitionDateTime',
]

csvlist = [
    "MagneticFieldStrength",
    "ManufacturersModelName",
    "PatientSex",
    "PatientAge",
    "PatientWeight",
]

def hashid(strid):
    return hashlib.md5(strid.encode("utf-8")).hexdigest()

def json_anon(infname, outfname):
    with open(infname,'r') as f:
        j = json.load(f)
        
        birthdate = j['PatientBirthDate']
        birthdate = datetime.fromisoformat(birthdate)
        studydate = j['AcquisitionDateTime'].split('T')[0]
        studydate = datetime.fromisoformat(studydate)
        patientage = np.round((studydate-birthdate).days/365.25)
        j['PatientAge'] = str(int(patientage))
        
        j['SeriesInstanceUID'] = hashid(j['SeriesInstanceUID'])
        j['StudyInstanceUID'] = hashid(j['StudyInstanceUID'])
        for key in anonlist:
            try:
                del j[key]
            except:
                pass
        with open(outfname,'w') as fo:
            json.dump(j, fo)
    return j

def deface(infname, outfname, facemask):
    nii = nb.load(infname)
    data = nii.get_fdata()
    if len(data.shape) == 4:
        data *= facemask[:,:,:,None]
    else:
        data *= facemask
    defaced = nb.Nifti1Image(data.astype(np.int16), header=nii.header, affine=nii.affine)
    defaced.header.set_data_dtype(np.int16)
    defaced.to_filename(outfname)


df1 = pd.read_csv("healthy_t1_t1ce_flairfs_t2_t2star_adc_tracew_mprage.csv")
df2 = pd.read_csv("patho_sample44_t1_t1ce_flairfs_t2_t2star_adc_tracew_mprage.csv")
df = pd.concat([df1,df2])

basepath = os.environ["HOME"] + "/data"
outpath = "./data_anonymized"

with open("samples_anon.csv", "w") as f:
    f.write(f"studyhash,processinghash,healthy,{','.join(csvlist)}\n")
    for i in range(len(df)):
        studyUID = df.iloc[i,1]   
        seriesUIDs = df.iloc[i,2:10]    
        uidhash = hashlib.md5("".join(seriesUIDs).encode("utf-8")).hexdigest()+"_t2"
        start_path = f'{basepath}/nii/{studyUID}'
        target_path = f'{outpath}/nii/{hashid(studyUID)}/{uidhash}'
        print(start_path)
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
        if i < len(df1):
            healthy="healthy"
        else:
            healthy="pathological"
            cmd = f"cp segment_patho/{studyUID}_seg.nii.gz {target_path}/mni_seg.nii.gz"
            os.system(cmd)
        f.write(f"{hashid(studyUID)},{uidhash},{healthy},")
        seqs = ["T1","T1CE","FLAIR","T2","T2S","ADC","TRACEW","MPRAGE"]
        for j, seriesUID in enumerate(seriesUIDs):
            jfname = f"{start_path}/{seriesUID}a.json"
            joutfname = f"{target_path}/{seqs[j]}.json"
            niiname = f"{start_path}/{uidhash}/{seqs[j]}mni.nii.gz"
            niioutname = f"{target_path}/{seqs[j]}mni.nii.gz"
            jstruct = json_anon(jfname, joutfname)
            if not os.path.isfile(niioutname):
                deface(niiname, niioutname, facemask)
        jsonvals = [str(jstruct[key]) for key in csvlist]
        f.write(",".join(jsonvals))
        f.write('\n')

df = pd.read_csv("samples_anon.csv")
df[:395].to_csv("samples_anon_healthy.csv", index=False)
df[395:].to_csv("samples_anon_patho.csv", index=False)