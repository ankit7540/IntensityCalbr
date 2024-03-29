
# Python module for vibration-rotation Raman intensities originating from common rotational states

+  Scheme for **vibration-rotation Raman intensities from H<sub>2</sub>, HD and D<sub>2</sub>.**
    + computing Raman intensities of sets of predefined transitions
    + use the computed intensities in a non-linear optimization analysis to determine the wavenumber dependent sensitivity
    + and, perform the above analysis using Raman intensity ratios from bands originating from the same rotational states.


# Usage
User supplied band area data arranged in 1D arrays are required. These should contain the band positions and experimental band intensities. Theoretical intensities will be computed within the iteration if temperature will be included as a fit parameter.

Requirements
----------------
Python 2.7 or Python 3.x with NumPy, SciPy and math modules


---

## Module details
The main modules are `genC2_CR_para` and `genC2_CR_perp`.


### Submodules


#####  `compute_series_para` & `compute_series_perp`

`compute_series_para` : to compute the parallel polarized relative Raman intensities of vibration-rotation transitions of H2, HD and D2

`compute_series_perp` : to compute the perpendicular polarized relative Raman intensities of vibration-rotation transitions of H2, HD and D2

`boltzmann_popln` : to compute the Boltzmann population of H2, HD and D2 for a given ro-vibrational state at a given temperature.

[See example]() for more details.
