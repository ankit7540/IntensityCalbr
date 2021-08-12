## Determining wavenumber-dependent sensitivity from Raman intensity ratio of the anti-Stokes and the corresponding Stokes band



This module uses known temperature (provided from user) and the laser frequency (in absolute wavenumbers) to compute the reference intensity ratio. This intensity ratio is compared with the experimental data and the wavenumber-dependent sensitivity is determined as a polynomial via least-squares minimization.


# Usage
User supplied band area data arranged as 2D arrays are required. These should contain the band positions and experimental band intensities. Theoretical intensity ratios will be computed within the program.

Requirements
----------------
Python 2.7 or Python 3.x with NumPy, SciPy and math modules. Matplotlib is required for plotting.

Example
----------------
[See example](https://github.com/ankit7540/IntensityCalbr/blob/master/PythonModule/determine_C2/vibrationalRaman_liquids/antiStokes_Stokes_ratios/example/example_antiStokes_Stokes_Raman_intensities.ipynb) for more details.
