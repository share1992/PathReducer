from dimensionality_reduction_functions import *


def get_pairs_of_atoms_given_distance_index(distance_indexes, atom_indexes):
    """
    Returns pairs of atom indices when given array of distance indices that comprise top distances in a PC
    :param distance_indexes: array
    :param atom_indexes: array
    :return: indexes_a, indexes_b: arrays
    """
    atom_indexes_a = []
    atom_indexes_b = []
    for w in range(np.array(distance_indexes).shape[0]):
        indexes_a_pcn = []
        indexes_b_pcn = []
        for n in distance_indexes[w]:
            i, j = calc_ij(n, len(atom_indexes))
            indexes_a_pcn.append(atom_indexes[i])
            indexes_b_pcn.append(atom_indexes[j])
        atom_indexes_a.append(indexes_a_pcn)
        atom_indexes_b.append(indexes_b_pcn)

    return atom_indexes_a, atom_indexes_b


def distance_coeffs(name, dim_num, indexes_a, indexes_b, dists_components):
    d = []
    for k in range(len(indexes_a)):
        coeff = dists_components[k]
        i = indexes_a[k]
        j = indexes_b[k]
        d.append({'atom 1': i, 'atom 2': j, 'Coefficient of Distance': coeff})

    d_df = pd.DataFrame(d)

    sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
    sorted_d = sorted_d.reset_index(drop=True)
    sorted_d.to_csv(name + "_output" + "/" + name + "_Ds_filtered" + '_PC%s_components.txt' % dim_num, sep='\t',
                    index=None)

    return sorted_d


def plot_key_distances(fignum, directory, name, n_dim, pc_dfs, points, distances, number_of_dists, red_or_full):
    fig = plt.figure(num=fignum, figsize=(16, 5))

    for n in range(n_dim):
        ax = plt.subplot(1, n_dim, n + 1)
        pc_df = pc_dfs[n]
        for i in range(number_of_dists):
            ax.scatter(points, distances[n][i], label='atoms %s and %s' % (pc_df['atom 1'][i], pc_df['atom 2'][i]))
            ax.set_xlabel('Point Number', fontsize=12)
            ax.set_ylabel('Distance (A)', fontsize=12)
            ax.legend()
            ax.set_title('Top Distances in PC%s' % (n+1))

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle("Key Distances in %s Dimensional Space" % red_or_full, fontsize=16)
    fig.savefig(directory + "/" + "%s_filtered_dists_in_%s_dim_space.png" % (name, red_or_full))


def distances_coeffs_calcij(directory, name, pc_num, pca_components, n):
    d = []
    for k in range(np.array(pca_components).shape[0]):
        i, j = calc_ij(k, n)
        coeff = pca_components[k]
        d.append({'atom 1': i, 'atom 2': j, 'Coefficient of Distance': coeff})

    d_df = pd.DataFrame(d)

    sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
    sorted_d.to_csv(directory + "/" + name + '_PC%s_components.txt' % pc_num, sep='\t', index=None)

    # Print results to console
    # print("Principal component %s:" % pc_num)
    # print(sorted_d)


def filter_top_distances(name, directory, n_dim, pca_components, full_dim_coords, red_dim_coords, n_top_atoms,
                         dist_threshold, number_of_dists=10):

    num_atoms = np.array(full_dim_coords).shape[1]
    print("Num atoms:", num_atoms)

    pca_components = np.reshape(pca_components, (pca_components.shape[0], int(pca_components.shape[1] / 3), 3))

    print("PCA Components shape:")
    print(np.array(pca_components).shape)

    approx_metrics = [np.sqrt(pca_components[n][:, 0] ** 2 + pca_components[n][:, 1] ** 2 +
                              pca_components[n][:, 2] ** 2) for n in range(n_dim)]
    print("Approx metrics shape:")
    print(np.array(approx_metrics).shape)

    # Determine top n atoms (default = 50) that make up the variance in principal component 1
    top_atom_indexes_1 = top_values(sum(approx_metrics[0:3]), n_top_atoms)
    print(top_atom_indexes_1)
    top_atom_indexes_1.sort()

    print(approx_metrics[0])
    print(top_atom_indexes_1)

    top_atom_indexes_2 = []
    dists = []
    for index_a in top_atom_indexes_1:
        for index_b in range(num_atoms):
            if index_b not in top_atom_indexes_1:
                # print(index_b)
                # Calculate only distances between atoms with the most variance and all other atoms. Only keep
                # distances less than x angstroms (default = 7.0) at the beginning (frame 0) of the trajectory
                a = full_dim_coords[0][index_a]
                b = full_dim_coords[0][index_b]
                d = np.linalg.norm(a - b)
                if d < dist_threshold and index_b not in top_atom_indexes_2:
                    # print("A: %s, B: %s, dist: %s" % (index_a, index_b, d))
                    dists.append(d)
                    top_atom_indexes_2.append(index_b)

    # Generate sorted list of indexes of atoms involved the most in PC1 and all atoms within the distance threshold
    top_atom_indexes = list(top_atom_indexes_1) + list(top_atom_indexes_2)
    top_atom_indexes.sort()

    # get_orig_atom_index = dict(zip(range(len(top_atom_indexes)), top_atom_indexes))

    # Calculate pairwise distances between atoms in top_indexes for each point along PC1
    # top_atom_distances = np.array([metrics.pairwise_distances(red_dim_coords[0, k, top_atom_indexes]) for k in range(red_dim_coords.shape[1])])
    # top_atom_coords = np.array(red_dim_coords[0:1, :, top_atom_indexes])
    # print(top_atom_coords.shape)
    # top_atom_coords = np.reshape(top_atom_coords, (top_atom_coords.shape[1], top_atom_coords.shape[2], top_atom_coords.shape[3]))
    # print(top_atom_coords.shape)
    # top_atom_distances = generate_ds(top_atom_coords)
    # print(top_atom_distances)
    top_atom_distances = np.array([metrics.pairwise_distances(red_dim_coords[0, k, top_atom_indexes]) for k in
                                   range(red_dim_coords.shape[1])])

    # Only use upper triangle of distance matrices
    reshaped_top_atom_dists = reshape_ds(top_atom_distances)

    # Do PCA on this reduced distance matrix
    matrix_pca_dists, matrix_pca_fit_dists, dists_components, dists_mean, dists_values, dists_x_1_2_3, dists_x_all = pca_dr(3,
                                                                                                                reshaped_top_atom_dists)

    print(dists_components)

    # All indexes
    ranked_distance_indexes = [top_values(abs(dists_components[n]), len(dists_components[n])) for n in range(n_dim)]

    print("Ranked distance indexes:")
    print(ranked_distance_indexes)

    top_distances_atom_indexes_1, top_distances_atom_indexes_2 = \
        get_pairs_of_atoms_given_distance_index(ranked_distance_indexes, top_atom_indexes)

    print(top_distances_atom_indexes_1, top_distances_atom_indexes_2)

    pc1_df = distance_coeffs(name, 1, top_distances_atom_indexes_1[0], top_distances_atom_indexes_2[0], dists_components[0])
    pc2_df = distance_coeffs(name, 2, top_distances_atom_indexes_1[1], top_distances_atom_indexes_2[1], dists_components[1])
    pc3_df = distance_coeffs(name, 3, top_distances_atom_indexes_1[2], top_distances_atom_indexes_2[2], dists_components[2])

    print("Top Distances for PC1-%s" % n_dim)
    print(pc1_df.head())
    print(pc2_df.head())
    print(pc3_df.head())

    # Plot top distances in X1-3 along original trajectory and in reduced dimensional space w.r.t. coordinates
    # Original traj, coordinates_all
    pc_dfs = (pc1_df, pc2_df, pc3_df)
    distances_full_PCs = []
    for pc_df in pc_dfs:
        distances = [np.linalg.norm(full_dim_coords[:, pc_df['atom 1'][i]] - full_dim_coords[:, pc_df['atom 2'][i]],
                                        axis=1) for i in range(len(pc_df))]
        distances_full_PCs.append(distances)


    # Reduced dimensional space w.r.t. coordinates, no_mass_weighting_xyz_coords_X_all
    red_dim_coords = np.reshape(red_dim_coords, (red_dim_coords.shape[1], red_dim_coords.shape[2],
                                                 red_dim_coords.shape[3]))
    distances_red_PCs = []
    for pc_df in pc_dfs:
        distances = [np.linalg.norm(red_dim_coords[:, pc_df['atom 1'][i]] - red_dim_coords[:, pc_df['atom 2'][i]],
                                    axis=1) for i in range(len(pc_df))]
        distances_red_PCs.append(distances)

    points = range(0, np.array(distances_full_PCs).shape[2])

    plot_key_distances(5, directory, name, n_dim, pc_dfs, points, distances_full_PCs, number_of_dists, "Full")
    plot_key_distances(6, directory, name, n_dim, pc_dfs, points, distances_red_PCs, number_of_dists, "Reduced")

    return pc1_df, pc2_df, pc3_df