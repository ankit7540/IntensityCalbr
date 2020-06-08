# IntensityCalbr

Repository containing programs implementing the combinatorial approach for obtaining wavelength dependent sensitivity from measured spectroscopic intensities when reference data is available. In this scheme, relative band intensities between all pairs of bands are analyzed simultaneously. Least squares minimization is used to determine the coefficients of a polynomial used to model the sensitivity.

## Methodology
Observed intensities from selected bands are analysed as pairs among all such bands, to form a matrix. A similar matrix of intensity ratios are compared to the true ratios, and the wavelength/wavenumber dependent sensitivity curve is obtained modelled as a polynomial function.

The general scheme is given as follows.

<img align="center" src="https://github.com/ankit7540/IntensityCalbr/blob/master/img/scheme.png" data-canonical-src="https://github.com/ankit7540/IntensityCalbr/blob/master/img/scheme.png" width="450" height="629" />

## References
In the following works, the ratio of intensities from common rotational states are compared to the corresponding theoretical ratio to obtain the wavelength dependent sensitivity curve.

  - H. Okajima, H. Hamaguchi, J. Raman Spectrosc. 2015, 46, 1140.
  - H. Hamaguchi, I. Harada, T. Shimanouchi, Chem. Lett. 1974, 3, 1405.

This principle of comparing intensities (rotational Raman and rotation-vibration Raman) is extended to all bands in present work, requiring paramterizing of temperature in the scheme. Set of intensity ratios are then conveniently framed as a matrix, as shown in the above figure. Refernce matrix can be computed if equations and required parameters are availble, or,  if known intensities are available then they can work as reference.


## Input data required

**Intensity calibration**

 - General scheme : experimental band area, reference data either available before hand or computable. (If computable then appropriate functions will be required to call). 
 
 In this work, compute code for intensities and reference matrix for pure rotation and rotational-vibrational Raman bands is given. (At present this is possible for H</sup>2</sup>, HD and D<sup>2</sup> since polarizability invariants are available for those.)

 - List of data required for analysis of pure rotational/ rotation-vibration Raman bands : experimental band area, xaxis vector for the spectra (in cm<sup>-1</sup> or wavelength). Indices of J-states for pure rotation; O,S,Q-bands for rotn-vibration bands and using these   reference data is computed on the fly.
 

See specific program's readme regarding the use of the above data.

## Usage

Clone the repository or download the zip file. As per your choice of the programming environment and refer to the specific README inside the folders and proceed.

## Comments

 - On convergence of the minimization scheme in intensity calibration : The convergence of the optimization has been tested with artificial and actual data giving expected results. However, in certain cases convergence in the minimization may not be achieved based on the specific data set and the error in the intensity.

 - Accuracy of the calibration : It is highly suggested to perform an independent validation of the intensity calibration. This validation can be using anti-Stokes to Stokes intensity for determining the sample's temperature (for checking the accuracy of wavelength sensitivity correction) and calculating the depolarization ratio from spectra (for checking the polarization dependent sensitivity correction). New ideas regarding testing the validity of intensity calibration are welcome. Please give comments in the "Issues" section of this repository.


## Credits
*Non-linear optimization in SciPy* :  Travis E. Oliphant. Python for Scientific Computing, Computing in Science & Engineering, 9, 10-20 (2007), DOI:10.1109/MCSE.2007.58


*Matplotlib*  : J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.


*Orthogonal Distance Regression as used in IgorPro and SciPy* : (***i***) P. T. Boggs, R. Byrd, R. Schnabel, SIAM J. Sci. Comput. 1987, 8, 1052. (***ii***) P. T. Boggs, J. R. Donaldson, R. h. Byrd, R. B. Schnabel, ACM Trans. Math. Softw. 1989, 15, 348. (***iii***) J. W. Zwolak, P. T. Boggs, L. T. Watson, ACM Trans. Math. Softw. 2007, 33, 27. (***iv***)  P. T. Boggs and J. E. Rogers, “Orthogonal Distance Regression,” in “Statistical analysis of measurement error models and applications: proceedings of the AMS-IMS-SIAM joint summer research conference held June 10-16, 1989,” Contemporary Mathematics, vol. 112, pg. 186, 1990.

## Support/Questions/Issues
Please use "Issues" section for asking questions and reporting issues.
