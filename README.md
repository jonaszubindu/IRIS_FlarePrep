# IRIS_FlarePrep

This repository is to prepare irisdata for machine learning. 
The parameters to each spectral line are defined in a spectral line dictionary. It contains all the necessary information to preprocess the spectral line. An example for the Magnesium II k&h line is given below:

MgIIk = {'lambda_min' : 2794,
         'lambda_max' : 2806,
         'n_breaks' : 960,
         'line' : "Mg II k",
         'field' : "NUV",
         'threshold' : 10
        }

The data is interpolated to a common wavelength grid according to the parameters in the line dict, calibrated to intensity in $erg s^{-1} cm^{-2} sr^{-1} \Angstrom^{-1}$ and filtered with the following steps:

1. Cosmic rays to zero. 
2. Pixel with -100 DN are set to zero.
3. Pixel with overexposure in 3 consecutive wavelength points are set to zero.
4. Pixels not reaching the minimum DN/s threshold are set to zero.
5. Timesteps during which the satellite passes through the southern atlantic anomaly SAA are set to zero.

The data is then normalized by its maximum value and the maximum values for each line are stored as well, to get intensities back.
Of the calibrated data in case of Mg II h&k we also use the following extra steps:

1. Maximum peak is inside a window close to the k or h core, as an extra step to remove cosmic rays in the pseudo-continuum range.
2. The peak ratio is within the theoretical window of $\[1,2\]$. Otherwise, there might be a cosmic ray or an error with the k or h core.
3. The pseudocontinuum level is below a relative limit of 40%

After these processing and calibration steps the data is stored in a class called Obs_raw_data. It is stored in the same structure as the raster data is stored. It has the three main attributes: 

1. times_global: array with the shape (2,number_of_exp). The first index contains the raster position under index 0 and the time under index 1.
2. im_arr_global: contains the normalized spectral data in the shape (time, y-pix, $\lambda$).
3. norm_vals_global: contains the maximum values from the normalization to get the intensities back. (time, y-pix, $\lambda$).

All other relevant data for each observation is stored as attributes of Obs_raw_data.
The data can then be used in the repository IRIScast to train neural networks at flare prediction and in the repository IRIS_AIA_HMI_align to create HMI/AIA IRIS aligned images.

