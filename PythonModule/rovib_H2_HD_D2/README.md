## Python module for vibration-rotation intensities

- Scheme for **vibration-rotation Raman intensities from H<sub>2</sub>, HD and D<sub>2</sub>.** This is the most detailed part of the repo, and includes functions for computing the true vibration-rotation Raman intensities for a given temperature.

For the determination of the wavenumber dependent sensitivity, the temperature can be either fixed or obtained by the analysis.

# Usage
User supplied band area data arranged in 1D arrays are required. These should contain the band positions and experimental band intensities. Theoretical intensities will be computed within the iteration if temperature will be included as a fit parameter.

Requirements
----------------
Python 2.7 or Python 3.x with NumPy, SciPy and math modules

Usage
----------------
Following commands are run under the Python interpreter environment. (Alternatively use any Python IDE like Spyder, IDLE or PyCharm). *Spyder3 is has been used while writing and debugging  python  codes given  here.*

***When using Python interpreter in terminal***

1. After cloning the repository and moving in the `python-module` directory,  refer to the readme.  Prepare the required data as mentioned above which will be loaded in the module  as numpy array. If required, change the path to the data files in the code.  

2. Import the python module. If  using python 2.7 add the current folder to path allowing to import the module in the current folder.

```
  import sys

  sys.path.append("..")
```

If using Python3, directly import as

 ```
    import wavelength_sensitivity
 ```

***When using Python IDE like Spyder***

1. After cloning the repository and moving in the `python-module` directory,  refer to the readme.  Prepare the required data as mentioned above which will be loaded in the module  as NumPy array. Open the  file in the IDE and make changes  to the file path if required. Run the code.
