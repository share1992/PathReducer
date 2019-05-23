import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_plot, colored_line_and_scatter_plot

# Number of PCA components
ndim = 3

# Input files and atom numbers surrounding stereogenic center
file = '/Users/ec18006/Documents/CHAMPS/Dimensionality_reduction/xyz_pdb_files/examples/neuraminidase_notH.xyz'

# DISTANCES INPUT. To mass-weight coordinates, add "MW=True" to function call.
system_name1, output_directory_D, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths1 = \
   dim_red.pathreducer(file, ndim, input_type="Distances")

# Plot results
D_pca_df = pd.DataFrame(D_pca)
colored_line_and_scatter_plot(D_pca_df[0], y=D_pca_df[1], y1=D_pca_df[2], same_axis=False, output_directory=output_directory_D,
          imgname=(system_name1 + "_Distances_noMW_scatterline"))

