# Process a bunch of trajectories in reduced dimensional space w.r.t. coordinates
def process_trajectories_coord_space(coordinates_pca_fit, directory):
    pca_components_trajectories = []
    reduced_dimensional_trajectories = []
    for file in list(glob.glob(directory + '*.xyz')):
        # Dimensionality reduction functions to transform xyz file into matrix input for PCA
        name_traj, energies_traj, atoms_traj, coordinates_all_traj = read_file(file)
        coordinates_shifted_traj = set_atom_one_to_origin(coordinates_all_traj)
        atom_masses_traj, mass_weighted_coords_traj = mass_weighting_pt(atoms_traj, coordinates_shifted_traj)
        processed_file = np.reshape(mass_weighted_coords_traj, (mass_weighted_coords_traj.shape[0],
                                                                mass_weighted_coords_traj.shape[2] *
                                                                mass_weighted_coords_traj.shape[1]))
        components = coordinates_pca_fit.transform(pd.DataFrame(processed_file))
        projected = coordinates_pca_fit.inverse_transform(pd.DataFrame(components))

        pca_components_trajectories.append(components)
        reduced_dimensional_trajectories.append(projected)

    return pca_components_trajectories, reduced_dimensional_trajectories

def distance_matrix_to_coords_alt(d):
    """ Converts a (2D square) distance matrix representation of a structure to Cartesian coordinates
    (first 3 columns correspond to 3D xyz coordinates) via a Gram matrix (DIFFERENT DEFINITION OF M).
    Should work exactly the same as distance_matrix_to_coords.
    :param d: 2D square matrix.
    :type d: numpy array
    :return: 3D Cartesian coordinates.
    :rtype: numpy array
    """

    d_one = np.reshape(d[:, 0], (d.shape[0], 1))

    m = (-0.5) * (d - np.matmul(np.ones((d.shape[0], 1)), np.transpose(d_one)) - np.matmul(d_one,
                                                                                           np.ones((1, d.shape[0]))))

    values, vectors = np.linalg.eig(m)

    idx = values.argsort()[::-1]
    values = values[idx]
    vectors = vectors[:, idx]

    assert np.allclose(np.dot(m, vectors), values * vectors)

    coords = np.dot(vectors, np.diag(np.sqrt(values)))

    # Only taking first three columns as Cartesian (xyz) coordinates
    coords = np.asarray(coords[:, 0:3])

    return coords


def calculate_mean(v):
    """ Calculates mean vector (in case of doing DR, all structures are centered around the mean).
    """
    mean_v = np.mean(v, axis=0)

    # Center all points around mean by explicitly subtracting mean distance matrix
    mean_centered_vs = []
    for i in range(len(v)):
        mean_centered_vs.append(v[i] - mean_v)

    mean_centered_vs = np.array(mean_centered_vs)
    return mean_v, mean_centered_vs


def reorient(start, atoms, coords):
    """Reorient regenerated structures via rotations and reflections (NOTE: THIS WILL AFFECT CHIRALITY).
    Kabsch algorithm is used to rotate structures to minimize RMSD.
    """
    start -= rmsd.centroid(start)
    axes_swap = []
    axes_reflect = []
    for i in range(len(coords)):
        coords[i] -= rmsd.centroid(coords[i])
        coords[i] = rmsd.kabsch_rotate(coords[i], start)
        min_rmsd, min_swap, min_reflection, min_review = rmsd.check_reflections(atoms, atoms, coords[i], start,
                                                                                reorder_method=None)
        axes_swap.append(min_swap)
        axes_reflect.append(min_reflection)
        coords[i][:, [0, 1, 2]] = coords[i][:, min_swap]
        coords[i] = coords[i] * min_reflection
        coords[i] = rmsd.kabsch_rotate(coords[i], coords[0])

    return coords


def chirality_changes(coords, a1, a2, a3, a4, negs=None, poss=None, signs=None):
    """ Determines chirality of structure and switches inconsistencies along a trajectory so chirality of the generated
    reduced dimensional IRC/trajectory is consistent with the FULL dimensional IRC/trajectory (i.e., the input).
    :param coords: xyz coordinates along IRC or trajectory, numpy array
    :param a1, a2, a3, a4: 4 atom numbers that represent groups around a chiral center, ints
    """
    if negs and poss and signs == None:
        negs, poss, signs = chirality_test(coords, a1, a2, a3, a4)

    if np.size(negs) > np.size(poss):
        print("Switching chirality of %s structures...\n" % np.size(poss))
        for i in range(np.size(poss)):
            loc = poss[0][i]
            # coords[loc] = -coords[loc]
            # Three coordinates switched
            test3 = coords
            test3[loc] = -test3[loc]
            if rmsd.kabsch_rmsd(coords[loc], coords[loc - 1]) > rmsd.kabsch_rmsd(test3[loc], coords[loc] - 1):
                coords[loc] = -coords[loc]

    if np.size(negs) < np.size(poss):
        print("Switching chirality of %s structures...\n" % np.size(negs))
        for i in range(np.size(negs)):
            loc = negs[0][i]
            # coords[loc] = -coords[loc]
            # Three coordinates switched
            test3 = coords
            test3[loc] = -test3[loc]
            if rmsd.kabsch_rmsd(coords[loc], coords[loc - 1]) > rmsd.kabsch_rmsd(test3[loc], coords[loc] - 1):
                coords[loc] = -coords[loc]

    return coords


# print(coords_comps.shape)
# atoms = range(0, int(len(coords_comps[0])/3))
# atoms = np.column_stack((atoms,atoms,atoms))
# atoms = np.reshape(atoms,(int(atoms.shape[0]*3),1))
# df = pd.DataFrame(coords_comps.T)
# df1 = pd.DataFrame(atoms)
# df = pd.concat([df, df1], axis=1)
# df.columns = ['PC1','PC2','PC3','Atom Index']
# coor = ['x', 'y', 'z']
# df['Cartesian Coordinate'] = pd.np.tile(coor, len(df) // len(coor)).tolist() + coor[:len(df)%len(coor)]
# print(df.head)
# df.to_csv(direc + "/" + system_name + 'coords_PC1_3_components.txt', sep='\t', index=None)