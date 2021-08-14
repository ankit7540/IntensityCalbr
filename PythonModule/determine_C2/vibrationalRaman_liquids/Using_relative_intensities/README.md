## Determining wavenumber-dependent sensitivity from Raman intensity ratio of vibrational transitions in selected liquids

This module uses pre-determined Raman intensity ratios of CCl<sub>4</sub>, C<sub>6</sub>H<sub>6</sub> and C<sub>6</sub>H<sub>12</sub> as the reference. This data is compared to the provided experimental data and the wavenumber-sensitivity is determined as a polynomial via least-squares minimization.

**Reference data, at room temperature (~298 K), taken from :**

  - Raj, A, Kato, C, Witek, HA, Hamaguchi, H. Toward standardization of Raman spectroscopy: Accurate wavenumber and intensity calibration using rotational Raman spectra of H<sub>2</sub>, HD, D<sub>2</sub>, and vibration–rotation spectrum of O<sub>2</sub>. <i>J Raman Spectrosc.</i> 2020; 51: 2066– 2082. https://doi.org/10.1002/jrs.5955

# Usage
User supplied band area data arranged as 2D arrays are required. These should contain the band positions and experimental band intensities. See sample data in the `example` folder for more details. Theoretical intensity ratios will be computed within the program.

Requirements
----------------
Python 2.7 or Python 3.x with NumPy, SciPy and math modules. Matplotlib is required for plotting.


Example
----------------
[See example](https://github.com/ankit7540/IntensityCalbr/blob/master/PythonModule/determine_C2/vibrationalRaman_liquids/Using_relative_intensities/example/example_genC2_from_vibration_Raman.ipynb) for more details.
