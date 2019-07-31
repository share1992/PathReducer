# _PathReducer_

_PathReducer_ is a tool that takes as input some set of molecular geometries in `.xyz` file format (e.g., an intrinsic reaction coordinate, a molecular dynamics trajectory, a relaxed potential energy surface scan) and outputs a reduced dimensional space for that set of molecular geometries. 

_PathReducer_ takes as input 
1. the path to the `.xyz` file (or files) of interest, as a string
2. the number of dimensions to reduce to, as an integer
3. optionally (though often necessary for visualization when representing the molecular structures as interatomic distances) the indexes of atoms (with numbering starting at 1, not 0) surrounding a stereogenic center in the system, as a list of integers
  
Currently, dimensionality reduction is conducted using Principal Component Analysis (PCA), though other dimensionality reduction techniques will be implemented in the future.

To initially define a reduced dimensional space, use the `pathreducer` function. To transform new data into an already defined reduced dimensional space, use the `transform_new_data` function.

## Installation
### Dependencies
Dependencies can be found in requirements.txt. To install all dependencies, use `pip install` in Terminal:

`pip install -r requirements.txt`

## Example Usage
Test scripts for using _PathReducer_'s basic functions are provided in the `test_scripts` folder. Additionally, the Jupyter notebook `PathReducer Walkthrough I.ipynb` guides you through an example system. An interactive function is also available to lead you through the process step-by-step by calling

`from dimensionality_reduction_functions import *`

`pathreducer_interactive()`

in a Python 3 shell. 
