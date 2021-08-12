## Determining wavenumber-dependent sensitivity from Raman intensity ratio of vibrational transitions in selected liquids

This module uses pre-determined Raman intensity ratios of CCl4, C6H6 and C6H12 as the reference. This data is compared to the provided experimental data and the wavenumber-sensitivity is determined as a polynomial via least-squares minimization.

**Reference data, at room temperature (~298 K), taken from :**

  - Raj, A, Kato, C, Witek, HA, Hamaguchi, H. Toward standardization of Raman spectroscopy: Accurate wavenumber and intensity calibration using rotational Raman spectra of H2, HD, D2, and vibration–rotation spectrum of O2. <i>J Raman Spectrosc.</i> 2020; 51: 2066– 2082. https://doi.org/10.1002/jrs.5955

# Usage
User supplied band area data arranged as 2D arrays are required. These should contain the band positions and experimental band intensities. Theoretical intensity ratios will be computed within the program.

Requirements
----------------
Python 2.7 or Python 3.x with NumPy, SciPy and math modules. Matplotlib is required for plotting.


Example
----------------
[See example](https://github.com/ankit7540/IntensityCalbr/blob/master/PythonModule/determine_C2/vibrationalRaman_liquids/Using_relative_intensities/example/example_genC2_from_vibration_Raman.ipynb) for more details.
