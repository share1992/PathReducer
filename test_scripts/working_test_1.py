import pandas as pd
import dimensionality_reduction_functions as dim_red
from lars_ddr import colorplot

################################################### Input Variables  ###################################################
file = 'examples/malondialdehyde/malondialdehydeTS_IRC.xyz'

# Number of PCA components
ndim = 3

# Atom indexes surrounding a stereogenic center
stereo_atoms_malon = [4, 7, 1, 8]

############################################### COORDINATES input to PCA ###############################################

# EXAMPLE 1B: Single file as input
traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, full, red = \
    dim_red.dr_routine(file, ndim, input_type="Coordinates")

# Plot trajectory in 2D and 3D
coords_pca_df = pd.DataFrame(coords_pca)
colorplot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, input_type="Coordinates",
          output_directory=direc, imgname=system_name)

################################################ DISTANCES input to PCA ################################################
# EXAMPLE 2B: Single file as input
traj_lengths1, system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = \
   dim_red.dr_routine(file, ndim, stereo_atoms=stereo_atoms_malon, input_type="Distances")

# Plot trajectory in 2D and 3D. Only specify "lengths=" if input to dr_routine was directory
D_pca_df = pd.DataFrame(D_pca)
colorplot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, input_type="Distances", output_directory=direc1,
          imgname=(system_name1 + "_D"))


coord1 = 'malondialdehydeTS_IRC_noMW_output/malondialdehydeTS_IRC_noMW_D_coord1.xyz'
name, energies, atoms, coordinates_all = dim_red.read_file(coord1)
deformation_PC1_R_TSS = dim_red.generate_deformation_vector(coordinates_all[0], coordinates_all[12])
deformation_PC1_TSS_P = dim_red.generate_deformation_vector(coordinates_all[12], coordinates_all[24])

coord2 = 'malondialdehydeTS_IRC_noMW_output/malondialdehydeTS_IRC_noMW_D_coord2.xyz'
name, energies, atoms, coordinates_all = dim_red.read_file(coord2)
deformation_PC2_R_TSS = dim_red.generate_deformation_vector(coordinates_all[0], coordinates_all[12])
deformation_PC2_TSS_P = dim_red.generate_deformation_vector(coordinates_all[12], coordinates_all[24])

coord3 = 'malondialdehydeTS_IRC_noMW_output/malondialdehydeTS_IRC_noMW_D_coord3.xyz'
name, energies, atoms, coordinates_all = dim_red.read_file(coord3)
deformation_PC3_R_TSS = dim_red.generate_deformation_vector(coordinates_all[0], coordinates_all[12])
deformation_PC3_TSS_P = dim_red.generate_deformation_vector(coordinates_all[12], coordinates_all[24])


