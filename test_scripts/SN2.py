import pandas as pd
import dimensionality_reduction_functions as dim_red
from lars_ddr import colorplot

# Number of PCA components
ndim = 3

#################################################### EXAMPLE 2: SN2 ####################################################
# Input files and atom numbers surrounding stereogenic center
file = 'examples/for_paper/SN2_all_for_paper.xyz'
stereo_atoms_SN2 = [1, 4, 5, 2]
new_file = 'examples/SN2/SN2_dyn_BOMD.xyz'

# CARTESIANS INPUT
traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, full, red = \
    dim_red.dr_routine(file, ndim, input_type="Cartesians", mass_weighting=True)
# Plot trajectory in 2D and 3D
coords_pca_df = pd.DataFrame(coords_pca)
colorplot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, input_type="Cartesians",
          output_directory=direc, imgname=system_name)
# Transforming new data into RD space
new_data_df = dim_red.transform_new_data(new_file, direc + "/new_data", ndim, coords_pca_fit,
              coords_comps, coords_mean, coords_pca, input_type="Cartesians", mass_weighting=True)
colorplot(coords_pca_df[0], y=coords_pca_df[1], y1=coords_pca_df[2], same_axis=False, new_data=new_data_df,
          input_type="Cartesians", output_directory=direc + "/new_data", imgname=system_name)

# DISTANCES INPUT
traj_lengths1, system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = \
   dim_red.dr_routine(file, ndim, stereo_atoms=stereo_atoms_SN2, input_type="Distances", mass_weighting=True)
# Plot trajectory in 2D and 3D
D_pca_df = pd.DataFrame(D_pca)
colorplot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, input_type="Distances", output_directory=direc1,
          imgname=(system_name1 + "_D"))
# Transforming new data into RD space
new_data_df = dim_red.transform_new_data(new_file, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                         D_pca, stereo_atoms=stereo_atoms_SN2, input_type="Distances", mass_weighting=True)
colorplot(D_pca_df[0], y=D_pca_df[1], y1=D_pca_df[2], same_axis=False, new_data=new_data_df, input_type="Distances",
          output_directory=direc1 + "/new_data", imgname=(system_name1 + "_D"))