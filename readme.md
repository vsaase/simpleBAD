Please note that the results presented in the paper were computed with the dataset prior to defacing.

The trained models can be downloaded from a server that is yet to be determined and will be published here.

The defaced dataset to train own models on can be requested from us and will be made available upon approval from our institutional ethics commitee.

To reproduce similar results with the defaced data follow these steps:
- contact us with a reasonable request to get access to the defaced dataset, place it in the folder data_anonymized
- install the required packages as given by requirements.txt
- run `python toh5_anon.py` to create the compressed healthy dataset x_z_anon.h5 for subsequent fast computations along all axes
- run `python inversecovariance -i x_z_anon.h5` to generate the baseline and covariance models, which are written to the file icov_x_z_anon.h5
- run `python vae.py --cuda --voxnorm --resblocks --mse --anon` to train the autoencoder. Model checkpoints are saved in the checkpoint folder, in this case the checkpoint used in the further analysis would be `vae_512_128_3_mse_rbvoxnorm.pt_best.pt`
- run `python test_healthy.py --anon` and `python test_patho.py --anon` to compute the residuals for all models, those are saved in the test_healthy and test_patho folders
- run `python voxelanalysis.py --anon` for computing the voxel performance metrics
- run `python sampleanalysis.py --anon` for computing the sample performance metrics
