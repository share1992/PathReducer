import pandas as pd
import numpy as np
import dimensionality_reduction_functions as dim_red
from plotting_functions import *

# Number of PCA components
ndim = 3

# Input files
system = '/Users/ec18006/Documents/Documents/CHAMPS/Explicit_solvent_MD/CHCl3_soln_1.xyz'

# DISTANCES INPUT
system_name, output_directory, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths, aligned_coords, \
blah, blah2 = dim_red.pathreducer(system, ndim, input_type="Distances", reconstruct=True, normal_modes=True)

D_pca_df = pd.DataFrame(D_pca)
colored_line_and_scatter_plot(D_pca_df[0], y=D_pca_df[1], y1=D_pca_df[2], output_directory=output_directory,
                              imgname=(system_name + "_Distances_noMW"))

