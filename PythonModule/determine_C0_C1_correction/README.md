# Generating C<sub>0</sub>/C<sub>1</sub> correction

The module `gen_correction.py` allows one to generate the correction (C<sub>0</sub>/C<sub>1</sub>)
using the function `gen_C0_C1`.
---
For a similar program written in IgorPro see [gen_correction.ipf](https://github.com/ankit7540/RamanSpec_BasicOperations/blob/master/intensity_corr/) in my other repository.
---
## Requirements
Python 3.6 or higher, numpy, scipy and Matplotlib


## Function

```
>>> import gen_correction
        **********************************************************

         This module is for generating the wavenumber-dependent
         intensity correction curve.

         Main function is listed below.

        **********************************************************
        gen_C0_C1 ( Ramanshift,  laser_nm, wl_spectra, norm_pnt,
                        mask = None, set_mask_nan = None, export = None)
        **********************************************************

         REQUIRED PARAMETERS
                         Ramanshift = vector, the x-axis in relative wavenumbers
                         laser_nm = scalar, the laser wavelength in nanometers
                         wl_spectra = broadband whitelight spectra (1D or 2D)
                         norm_pnt =  normalization point (corrections will be set
                                        to unity at this point
          OPTIONAL PARAMETERS
                         mask = vector, mask wave for selecting specific region to fit
                         set_mask_nan= boolean, 1 will set the masked region in
                                        the output correction to nan, 0 will not do so.
                         export = 0 or 1, setting to 1 will export the correction as a txt
                                     file with name intensity_correction.txt
                          ------------------------------------------
                          All vectors required here should be numpy arrays.
                          See line 14 to 18 to define/load the numpy arrays
                                              before execution.
        **********************************************************
        ```
