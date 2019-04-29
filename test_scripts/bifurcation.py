import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_plot

# Number of PCA components
ndim = 3

####################################### EXAMPLE 4: CYCLOPROPYLIDENE BIFURCATION ########################################
# Inputs
file = '../examples/bifurcation/bifur_IRC.xyz'
stereo_atoms_B = [4, 5, 6, 7]
# "New Files" to test transforming trajectories into already generated reduced dimensional space
new_file1 = '../examples/bifurcation/bifur_traj1.xyz'
new_file2 = '../examples/bifurcation/bifur_traj2.xyz'
new_file3 = '../examples/bifurcation/bifur_traj3.xyz'
new_file4 = '../examples/bifurcation/bifur_traj4.xyz'

# CARTESIANS INPUT
traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals = \
    dim_red.dr_routine(file, ndim, input_type="Cartesians")

# DISTANCES INPUT
traj_lengths1, system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = dim_red.dr_routine(file,
                                                            ndim, stereo_atoms=stereo_atoms_B, input_type="Distances")

# Transforming new data into RD space
new_data_df1 = dim_red.transform_new_data(new_file1, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                         D_pca, stereo_atoms=stereo_atoms_B, input_type="Distances")
new_data_df2 = dim_red.transform_new_data(new_file2, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                         D_pca, stereo_atoms=stereo_atoms_B, input_type="Distances")
new_data_df3 = dim_red.transform_new_data(new_file3, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                         D_pca, stereo_atoms=stereo_atoms_B, input_type="Distances")
new_data_df4 = dim_red.transform_new_data(new_file4, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                         D_pca, stereo_atoms=stereo_atoms_B, input_type="Distances")


# Plotting
# CARTESIANS INPUT
coords_pca_df = pd.DataFrame(coords_pca)
coords_pca_df1 = coords_pca_df[0:183]
coords_pca_df2 = coords_pca_df.drop(coords_pca_df.index[106:184], axis=0)
colorplot(coords_pca_df1[0], y=coords_pca_df1[1], y1=coords_pca_df1[2], x2=coords_pca_df2[0], y2=coords_pca_df2[1],
          y12=coords_pca_df2[2], same_axis=False, output_directory=direc, imgname=system_name)

# DISTANCES INPUT
D_pca_df = pd.DataFrame(D_pca)
D_pca_df1 = D_pca_df[0:183]
D_pca_df2 = D_pca_df.drop(D_pca_df.index[106:184], axis=0)
colored_line_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, output_directory=direc1, imgname=(system_name1 + "_D"))

colored_line_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, new_data=new_data_df1, output_directory=direc1 + "/new_data",
          imgname=(system_name1 + "_D_noMW_traj1"))
colored_line_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, new_data=new_data_df2, output_directory=direc1 + "/new_data",
          imgname=(system_name1 + "_D_noMW_traj2"))
colored_line_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, new_data=new_data_df3, output_directory=direc1 + "/new_data",
          imgname=(system_name1 + "_D_noMW_traj3"))
colored_line_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, new_data=new_data_df4, output_directory=direc1 + "/new_data",
          imgname=(system_name1 + "_D_noMW_traj4"))
