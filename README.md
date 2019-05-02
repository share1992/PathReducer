# PathReducer

PathReducer is a tool that takes as input some set of molecular geometries in `.xyz` file format (e.g., an intrinsic reaction coordinate, a molecular dynamics trajectory, a relaxed potential energy surface scan) and outputs a reduced dimensional space for that set of molecular geometries. 

PathReducer takes as input 
1. the path to the `.xyz` file (or files) of interest, as a string
2. the number of dimensions to reduce to, as an integer
3. optionally (though often necessary for visualization when representing the molecular structures as interatomic distances) the indexes of atoms (with numbering starting at 1, not 0) surrounding a stereogenic center in the system, as a list of integers
  
Currently, dimensionality reduction is conducted using Principal Component Analysis (PCA), though other dimensionality reduction techniques will be implemented in the future.

PathReducer outputs...

To transform new data into an already defined reduced dimensional space, use the "transform_new_data" function.

## Installation
### Dependencies
Pathreducer: numpy, pandas, math, glob, os, matplotlib, sklearn, periodictable, calculate_rmsd as rmsd
Plotting functions: numpy, matplotlib, scipy, mpl_toolkits, os

## Example Usage
`file = 'examples/reaction_1/reaction_coordinate.xyz'`  
`ndim = 3`    
`stereo_atoms = [1, 2, 3, 4]`      

### Cartesian coordinates input to PCA
`system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, traj_lengths = \
    dim_red.dr_routine(files, ndim, input_type="Cartesians")`

### Interatomic distances input to PCA
`system_name, output_directory, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths = \
   dim_red.dr_routine(files, ndim, stereo_atoms=stereo_atoms, input_type="Distances")`
   
### Transforming new data into reduced dimensional space
`new_data_PCs = dim_red.transform_new_data(new_file, output_directory + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean, D_pca, stereo_atoms=stereo_atoms, input_type="Distances")`
