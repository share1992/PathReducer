import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_and_scatter_plot, colored_line_plot_projected_data, colored_scatter_plot
import matplotlib.pyplot as plt

# Number of PCA components
ndim = 3

############################################### EXAMPLE 1: MALONALDEHYDE ###############################################
# Input file and list of atom numbers surrounding stereogenic center
input_path = '/Users/ec18006/OneDrive - University of Bristol/CHAMPS/Research_Topics/cyclopropylidene_bifurcation/including_N2/optimizations_and_IRCs/complete_TSS_cycN2_b3lyp_left_and_right_IRC.xyz'
new_file1 = '/Users/ec18006/OneDrive - University of Bristol/CHAMPS/Research_Topics/cyclopropylidene_bifurcation/including_N2/gas_phase_MD/progdyn_trajectories/n1/traj4.xyz'
energies = '/Users/ec18006/OneDrive - University of Bristol/CHAMPS/Research_Topics/cyclopropylidene_bifurcation/including_N2/optimizations_and_IRCs/TSS_cycN2_b3lyp_left_and_right_energies.txt'


# DISTANCES INPUT, NOT MASS-WEIGHTED. To mass-weight coordinates, add "MW=True" to function call.
# system_name, output_directory, pca, pca_fit, pca_components, mean, values, lengths, aligned_original_coords = \
#     dim_red.pathreducer(input_path, ndim, input_type="Distances")
system_name, output_directory, pca, pca_fit, pca_components, mean, values, lengths, aligned_original_coords = \
    dim_red.pathreducer(input_path, ndim, input_type="Distances")

# Plot results
pca_df = pd.DataFrame(pca)
x = pca_df[0]
y = pca_df[1]
z = pca_df[2]

new_name1, new_data_df1 = dim_red.transform_new_data(new_file1, output_directory + "/new_data", ndim, pca_fit, pca_components, mean,
                                          aligned_original_coords, input_type="Distances")

x_new = new_data_df1[0]
y_new = new_data_df1[1]
z_new = new_data_df1[2]

# Plotting new data into defined RD space
time = list(range(len(x)))
time_new = list(range(len(x_new)))

x = list(x)
y = list(y)
z = list(z)

x_new = list(x_new)
y_new = list(y_new)
z_new = list(z_new)

# Create figure
fig0 = plt.figure(figsize=(6, 5))
ax0 = fig0.add_subplot(1, 1, 1)
ax0.grid(True)

ax0.scatter(x, y, s=100, c=time, cmap='viridis', edgecolors='k')
ax0.scatter(x_new, y_new, s=30, c=time_new, cmap='viridis')

x_all = x + x_new
y_all = y + y_new
z_all = z + z_new

ax0.set_xlabel("PC1", fontsize=16)
ax0.set_ylabel("PC2", fontsize=16)
ax0.set_xlim(min(x_all) - 0.1 * max(x_all), 1.1 * max(x_all))
ax0.set_ylim(min(y_all) - 0.1 * max(y_all), 1.1 * max(y_all))
ax0.tick_params(labelsize=14, pad=10)

fig1 = plt.figure(figsize=(7, 6))
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.grid(True)

ax1.scatter(x, y, z, s=100, c=time, cmap='viridis', edgecolors='k')
ax1.scatter(x_new, y_new, z_new, s=30, c=time_new, cmap='viridis')

ax1.set_xlabel("PC1", fontsize=16, labelpad=20)
ax1.set_ylabel("PC2", fontsize=16, labelpad=20)
ax1.set_zlabel("PC3", fontsize=16, labelpad=20)
ax1.ticklabel_format(style='sci', scilimits=(-4, 4))

ax1.set_xlim(min(x_all) - 0.1 * max(x_all), 1.1 * max(x_all))
ax1.set_ylim(min(y_all) - 0.1 * max(y_all), 1.1 * max(y_all))
ax1.tick_params(labelsize=14, pad=10)

# ax1.view_init(elev=20., azim=60)

fig2 = plt.figure(figsize=(7, 6))
ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
ax2.grid(True)

energy_table = pd.read_csv(energies, delim_whitespace=True, header=None)
energies = energy_table[4]
relative_energies = (energy_table[4] - min(energy_table[4]))*627.51
ax2.scatter(x, y, relative_energies, s=100, c=time, cmap='viridis', edgecolors='k')

plt.show()


