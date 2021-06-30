# IntensityCalbr

Repository : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4655294.svg)](https://doi.org/10.5281/zenodo.4655294)

Repository containing programs implementing the procedure for obtaining wavelength-dependent sensitivity for calibration of Raman spectrometers based on multi-channel detectors. The present scheme is a multi-step procedure based on following three steps:
- C<sub>0</sub> : Correction for non-linear sampling of photons in the wavenumber scale.
- C<sub>1</sub> : Correction for channel-to-channel variation in the sensitivity of the spectrometer.
- C<sub>2</sub> : Final correction derived from Raman spectroscopic intensities.

In order to determine the final correction (C<sub>2</sub>) the relative band intensities between all pairs of bands are analyzed simultaneously by a comparison with the analogous reference intensities. Least squares minimization is used to determine the coefficients of a polynomial used to model the wavelength-dependent sensitivity representing the C<sub>2</sub> correction.

## Methodology
Observed intensities from selected bands are analyzed as pairs among all such bands, to form a matrix. A similar matrix of intensity ratios are compared to the true ratios, and the coefficients for the wavelength/wavenumber dependent sensitivity curve, modelled as a polynomial function, is obtained via non-linear minimization technique.

The general scheme is given as follows.

<p align="center">
<img  src="https://github.com/ankit7540/IntensityCalbr/blob/master/img/scheme.png" data-canonical-src="https://github.com/ankit7540/IntensityCalbr/blob/master/img/scheme.png" width="450" height="629" /> </p>

Explanation for the steps of the scheme are following :


-  The experimental data available as 2D array is used to generate the **R**<sub>obs.</sub> matrix. Using the errors in band areas, the weights are generated.
-  The reference data computed at the given temperature is used to generate the **R**<sub>true</sub> matrix.
-  Next, using the band positions and initial coefs of the polynomial, the  **S**  matrix is generated.
-  The dimensions of the four matrices are checked before moving to the next step.
-  Difference matrix, **D**<sub></sub>, (for each species) is generated using the **R**<sub>obs</sub>, **R**<sub>true</sub> and **S**  matrix.
-  The elements of the difference matrix are weighted using the corresponding elements of the weight matrix **W**.
-  The norm of the difference matrix is computed. This norm is minimized by varying the coefficients of the polynomial (recomputing the  **S**  matrix and the reference matrix, **R**<sub>true</sub> using the temperature ).
-  Use the optimized coefficients of the polynomial to generate the C<sub>2</sub> correction. Check temperature obtained from the Raman intensities for physical correctness.


## References
In the following works, the ratio of intensities from common rotational states are compared to the corresponding theoretical ratio to obtain the wavelength dependent sensitivity curve.

  - H. Okajima, H. Hamaguchi, J. Raman Spectrosc. 2015, 46, 1140.
  - H. Hamaguchi, I. Harada, T. Shimanouchi, Chem. Lett. 1974, 3, 1405.
  - A. Raj, C. Kato, H.A. Witek. H. Hamaguchi, J. Raman Spec 2020 (submitted)

This principle of comparing intensities (pure rotational Raman and rotation-vibration Raman) is extended to all bands in present work, requiring parametrizing of temperature in the scheme. Set of intensity ratios are then conveniently framed as a matrix, as shown in the above figure. The reference matrix can be computed if equations and required parameters are available, or,  if known intensities are available then they can work as reference (for given conditions).


## Input data required

**Intensity calibration**

 - Determination of C<sub>0</sub> and C<sub>1</sub> requires the vector/array of relative wavenumbers (which is used as the x-axis) and the measured spectrum of a broadband white-light source (we assume here that this source is close to a black-body emitter, so tungsten lamps will work).

 - General scheme : experimental band area, reference data either available before hand or computable. (If computable then appropriate functions are required to be called).

 In this work, compute code for intensities and reference matrix for pure rotation and rotational-vibrational Raman bands are given. (At present this is possible for H<sub>2</sub>, HD and D<sub>2</sub> since polarizability invariants are available for these from our earlier work [See https://doi.org/10.1063/1.5011433 ].)

 - List of data required for analysis of pure rotational/ rotation-vibration Raman bands : experimental band area, x-axis vector for the spectra (in cm<sup>-1</sup> or wavelength). Indices of J-states for pure rotation; O,S,Q-bands for rotational-vibration bands, temperature (K) as additional parameters. The reference data is computed on the fly.


See specific program's readme regarding the use of the above data.

## Usage

Clone the repository or download the zip file. As per your choice of the programming environment and refer to the specific README inside the folders and proceed.

## Comments

 - **On convergence of the minimization scheme in intensity calibration :** The convergence of the optimization has been tested with artificial and actual data giving expected results. However, in certain cases convergence in the minimization may not be achieved based on the specific data set and the error in the intensity.

 - **Accuracy of the calibration :** It is highly suggested to perform an independent validation of the intensity calibration. This validation can be using anti-Stokes to Stokes intensity for determining the sample's temperature (for checking the accuracy of wavelength sensitivity correction). New ideas regarding testing the validity of intensity calibration are welcome. Please give comments in the "Issues" section of this repository.


## Credits
*Non-linear optimization in SciPy* :  Travis E. Oliphant. Python for Scientific Computing, Computing in Science & Engineering, 9, 10-20 (2007), DOI:10.1109/MCSE.2007.58


*Matplotlib*  : J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.


*Orthogonal Distance Regression as used in IgorPro and SciPy* : (***i***) P. T. Boggs, R. Byrd, R. Schnabel, SIAM J. Sci. Comput. 1987, 8, 1052. (***ii***) P. T. Boggs, J. R. Donaldson, R. h. Byrd, R. B. Schnabel, ACM Trans. Math. Softw. 1989, 15, 348. (***iii***) J. W. Zwolak, P. T. Boggs, L. T. Watson, ACM Trans. Math. Softw. 2007, 33, 27. (***iv***)  P. T. Boggs and J. E. Rogers, “Orthogonal Distance Regression,” in “Statistical analysis of measurement error models and applications: proceedings of the AMS-IMS-SIAM joint summer research conference held June 10-16, 1989,” Contemporary Mathematics, vol. 112, pg. 186, 1990.

## Support/Questions/Issues
Please use "Issues" section for asking questions and reporting issues.
