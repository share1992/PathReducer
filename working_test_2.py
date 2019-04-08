import pandas as pd
import dimensionality_reduction_functions as dim_red
from lars_ddr import colorplot

################################################### Input Variables  ###################################################

file = 'examples/bifurcations/bifur.xyz'

# "New Files" to test transforming trajectories into already generated reduced dimensional space
new_file = 'examples/bifurcations/traj1.xyz'
# new_file = 'examples/bifurcations/traj2.xyz'
# new_file = 'examples/bifurcations/traj3.xyz'
# new_file = 'examples/bifurcations/traj4.xyz'

# Number of PCA components
ndim = 3

# Barry's bifurcation system atoms attached to allene core
B_stereo_atoms = [5, 6, 7, 8]

############################################### COORDINATES input to PCA ###############################################

traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, full, red = \
    dim_red.dr_routine(file, ndim, input_type="Coordinates")

# Transforming new data into generated reduced dimensional space
new_data_df = dim_red.transform_new_data(new_file, direc + "/new_data", ndim, coords_pca_fit, coords_comps, coords_mean,
                                         coords_pca, input_type="Coordinates")

# Bifurcation stuff: Two dataframes are made so that the coloring in color plot shows two paths ending in yellow
coords_pca_df = pd.DataFrame(coords_pca)
coords_pca_df1 = coords_pca_df[0:183]
coords_pca_df2 = coords_pca_df.drop(coords_pca_df.index[106:184], axis=0)
colorplot(coords_pca_df1[0], y=coords_pca_df1[1], y1=coords_pca_df1[2], x2=coords_pca_df2[0], y2=coords_pca_df2[1],
          y12=coords_pca_df2[2], same_axis=False, input_type="Coordinates", output_directory=direc, imgname=system_name)

# MD trajectory data being projected into RD space
colorplot(coords_pca_df1[0], y=coords_pca_df1[1], y1=coords_pca_df1[2], x2=coords_pca_df2[0], y2=coords_pca_df2[1],
          y12=coords_pca_df2[2], same_axis=False, new_data=new_data_df, input_type="Coordinates",
          output_directory=direc + "/new_data", imgname=system_name)

################################################ DISTANCES input to PCA ################################################

traj_lengths1, system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = \
   dim_red.dr_routine(file, ndim, stereo_atoms=B_stereo_atoms, input_type="Distances")

# Bifurcation stuff: Two dataframes are made so that the coloring in color plot shows two paths ending in yellow
D_pca_df = pd.DataFrame(D_pca)
D_pca_df1 = D_pca_df[0:183]
D_pca_df2 = D_pca_df.drop(D_pca_df.index[106:184], axis=0)
colorplot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, input_type="Distances", output_directory=direc1, imgname=(system_name1 + "_D"))

# MD trajectory data being projected into RD space
new_data_df = dim_red.transform_new_data(new_file, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                         D_pca, stereo_atoms=B_stereo_atoms, input_type="Distances")
colorplot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, new_data=new_data_df, input_type="Distances", output_directory=direc1 + "/new_data",
          imgname=(system_name1 + "_D_noMW_traj1"))
