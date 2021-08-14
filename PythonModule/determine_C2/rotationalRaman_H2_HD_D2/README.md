# Python module for generating C<sub>2</sub> correction from pure rotational Raman intensities

**This module is for:**

 - Scheme for the **pure rotational Raman intensities from H<sub>2</sub>, HD and D<sub>2</sub>.** This includes functions for computing the true intensities for a given temperature. If temperature is not needed as a fit variable then computation of spectra at some fixed temperature is also possible.

# Usage
User supplied band area data arranged as 2D arrays. These should contain the band positions and experimental band intensities. Theoretical intensities will be computed within the iteration if temperature will be included as a fit parameter. See sample data in `example` directory.

Requirements
----------------
Python 2.7 or Python 3.x with NumPy, SciPy and math modules


## Temperature dependent analysis

This analysis has temperature as a fit parameter, which is optimized during the analysis along with the coefficients used to model the C<sub>2</sub> correction. [(See here)](https://github.com/ankit7540/IntensityCalbr/tree/master/PythonModule/determine_C2/rotationalRaman_H2_HD_D2/t_dependent)


## Temperature fixed analysis

This analysis has temperature as a fixed parameter, which is provided by the user and doesn't change during the analysis. [(See here)](https://github.com/ankit7540/IntensityCalbr/tree/master/PythonModule/determine_C2/rotationalRaman_H2_HD_D2/t_independent)
