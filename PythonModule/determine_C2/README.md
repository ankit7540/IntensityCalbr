## Python modules for determining C<sub>2</sub> correction from Raman intensities

The modules include.

 - Scheme for the pure rotational Raman intensities from H<sub>2</sub>, HD and D<sub>2</sub>. This includes functions for computing the true intensities for a given temperature. If temperature is not needed as a fit variable then computation of spectra at some fixed temperature is also possible. Refer to the directory `rotationalRaman_H2_HD_D2` directory and the various sub-directories within.  

 - Scheme for vibration-rotation Raman intensities from H<sub>2</sub>, HD and D<sub>2</sub>. This is the most detailed part of the repo, and includes functions for computing the true intensities for a given temperature and further analysis based on these intensities. If temperature is not needed as a fit variable then computation of spectra at some fixed temperature is also possible. Refer to the directory `vibration_rotation_H2_HD_D2` directory and the various sub-directories within.

 - Scheme for determining wavenumber dependent sensitivity from Raman spectra of liquids. Two available techniques are : i) using reference intensity ratios determined from prior work, and ii) anti-Stokes to Stokes intensity ratio at a given temperature.

*Checking examples and sample data for each technique is highly suggested.*

# Usage
User supplied band area data arranged in 1D arrays are required. These should contain the band positions and experimental band intensities. Theoretical intensities will be computed within the program. These intensities will be updated during each iteration if temperature is included as a fit parameter.

Requirements
----------------
Python 2.7 or Python 3.x with NumPy, SciPy and math modules. Matplotlib is required for plotting.

Usage
----------------
Following commands are run under the Python interpreter environment. (Alternatively use any Python IDE like Spyder, IDLE or PyCharm). *Spyder3 is has been used while writing and debugging  python  codes given  here.*

***When using Python interpreter in terminal***

1. After cloning the repository and moving in the `python-module` directory,  refer to the `readme`.  Prepare the required data as mentioned above which should loaded in the module as numpy array. This would require changing the path to the data files in the script.  

2. Import the python module. If  using python 2.7 add the current folder to path allowing to import the module in the current folder.

```
   import sys

   sys.path.append("..")
```

If using Python3, directly import as

  ```
     import <file_name>
  ```

***When using Python IDE like Spyder***

After cloning the repository and moving in the `PythonModule` directory,  refer to the readme.  Prepare the required data as mentioned above which should be loaded as NumPy array(s). Open the  file in the IDE and make changes to the file path if required. Run the code.
