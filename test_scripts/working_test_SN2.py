import pandas as pd
import dimensionality_reduction_functions as dim_red
from lars_ddr import colorplot

################################################### Input Variables  ###################################################
file = 'examples/SN2/SN2_all.xyz'
new_file = 'examples/SN2/SN2_dyn_BOMD.xyz'

# Number of PCA components
ndim = 3

# Atom indexes surrounding a stereogenic center
stereo_atoms_SN2 = [1, 4, 5, 2]

############################################### COORDINATES input to PCA ###############################################
#
# # EXAMPLE 1B: Single file as input
# traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, full, red = \
#     dim_red.dr_routine(file, ndim, input_type="Coordinates")
#
# # Plot trajectory in 2D and 3D
# coords_pca_df = pd.DataFrame(coords_pca)
# colorplot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, input_type="Coordinates",
#           output_directory=direc, imgname=system_name)

################################################ DISTANCES input to PCA ################################################
# EXAMPLE 2B: Single file as input
traj_lengths1, system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = \
   dim_red.dr_routine(file, ndim, stereo_atoms=stereo_atoms_SN2, input_type="Distances")

# Plot trajectory in 2D and 3D. Only specify "lengths=" if input to dr_routine was directory
D_pca_df = pd.DataFrame(D_pca)
colorplot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, input_type="Distances", output_directory=direc1,
          imgname=(system_name1 + "_D"))

# Transforming new data into generated reduced dimensional space
new_data_df = dim_red.transform_new_data(new_file, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                         D_pca, stereo_atoms=stereo_atoms_SN2, input_type="Distances")
colorplot(D_pca_df[0], y=D_pca_df[1], y1=D_pca_df[2], same_axis=False, new_data=new_data_df, input_type="Distances",
          output_directory=direc1 + "/new_data", imgname=(system_name1 + "_D"))


coord1 = 'SN2_all_noMW_output/SN2_all_noMW_D_coord1.xyz'
name, energies, atoms, coordinates_all1 = dim_red.read_file(coord1)
deformation_PC1_A_B = dim_red.generate_deformation_vector(coordinates_all1[0], coordinates_all1[15])
deformation_PC1_B_C = dim_red.generate_deformation_vector(coordinates_all1[15], coordinates_all1[50])
deformation_PC1_C_D = dim_red.generate_deformation_vector(coordinates_all1[50], coordinates_all1[103])

coord2 = 'SN2_all_noMW_output/SN2_all_noMW_D_coord2.xyz'
name, energies, atoms, coordinates_all2 = dim_red.read_file(coord2)
deformation_PC2_A_B = dim_red.generate_deformation_vector(coordinates_all2[0], coordinates_all2[15])
deformation_PC2_B_C = dim_red.generate_deformation_vector(coordinates_all2[15], coordinates_all2[50])
deformation_PC2_C_D = dim_red.generate_deformation_vector(coordinates_all2[50], coordinates_all2[103])

coord3 = 'SN2_all_noMW_output/SN2_all_noMW_D_coord3.xyz'
name, energies, atoms, coordinates_all3 = dim_red.read_file(coord3)
deformation_PC3_A_B = dim_red.generate_deformation_vector(coordinates_all3[0], coordinates_all3[15])
deformation_PC3_B_C = dim_red.generate_deformation_vector(coordinates_all3[15], coordinates_all3[50])
deformation_PC3_C_D = dim_red.generate_deformation_vector(coordinates_all3[50], coordinates_all3[103])
