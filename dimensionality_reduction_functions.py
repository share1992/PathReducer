#!/usr/bin/env python
# coding: utf-8


import numpy as np
import calculate_rmsd as rmsd
import pandas as pd
import math
import glob
import os
from periodictable import *
from matplotlib import pyplot as plt
from sklearn import *
from lars_ddr import colorplot
from filter_top_distances import filter_top_distances


def read_file(f):
    """ Reads in each file, and for each file, separates each IRC point into its own matrix of cartesian coordinates.
    coordinates_all is arranged as coordinates_all[n][N][c], where n is the IRC point, N is the atom number, and c
    is the x, y, or z coordinate.
    """
    print("File being read is: %s" % f)
    name = f.split('/')[-1].split('.')[-2]

    xyz = open(f)
    n_atoms = int(xyz.readline())
    energies = []
    atoms = []
    coordinates = []

    # Each point along the IRC/traj will be stored as an entry in a 3D array called coordinates_all
    coordinates_all = []
    for line in xyz:
        splitline = line.split()
        if len(splitline) == 4:
            atom, x, y, z = line.split()
            atoms.append(atom)
            coordinates.append([float(x), float(y), float(z)])
        elif len(splitline) == 1:
            if type(splitline[0]) == str:
                pass
            elif float(splitline[0]) == float(n_atoms):
                pass
            else:
                energy = float(splitline[0])
                energies.append(energy)
            if len(coordinates) != 0:
                coordinates_all.append(coordinates)
            elif len(coordinates) == 0:
                pass
            atoms = []
            coordinates = []
    else:
        coordinates_all.append(coordinates)

    xyz.close()

    coordinates_all = np.asarray(coordinates_all)

    # Print ERROR if length of coordinate section doesn't match number of atoms specified at beginning of xyz file
    if len(atoms) != n_atoms:
        print("ERROR: file contains %d atoms instead of the stated number %d" % (n_atoms, len(atoms)))
        print("number of atoms in file: %d" % len(atoms))
        print("number of coordinates:   %d" % len(coordinates))

    return name, energies, atoms, coordinates_all


def set_atom_one_to_origin(coordinates):
    coordinates_shifted = coordinates - coordinates[:, np.newaxis, 0]

    return coordinates_shifted


def mass_weighting_pt(atoms, coordinates):
    atom_masses = []
    for atom in atoms:
        atom_mass = formula(atom).mass
        atom_masses.append(atom_mass)

    weighting = np.sqrt(atom_masses)
    weighting_tri = np.column_stack((weighting, weighting, weighting))

    mass_weighted_coords = coordinates * weighting_tri[np.newaxis, :, :]

    return atom_masses, mass_weighted_coords


def unmass_weighting_pt(atoms, coordinates):
    atom_masses = []
    for atom in atoms:
        atom_mass = formula(atom).mass
        atom_masses.append(atom_mass)

    weighting = np.sqrt(atom_masses)
    weighting_tri = np.column_stack((weighting, weighting, weighting))

    unmass_weighted_coords = coordinates / weighting_tri[np.newaxis, :, :]

    return unmass_weighted_coords


def generate_ds(coordinates):
    """ Generates distance matrices, either for each structure or between structures, depending on the input.
    """
    d2 = np.sum((coordinates[:, :, None] - coordinates[:, None, :]) ** 2, axis=3)
    return d2


def reshape_ds(d):
    """ Takes only the upper triangle of the distance matrices and reshapes them into 1D arrays.
    """
    d_re = []
    x = d[0][0].shape[0]

    for dint in d:
        dint_re = dint[np.triu_indices(x, k=1)]
        d_re.append(dint_re)

    d_re = np.asarray(d_re)

    return d_re


def vector_to_matrix(v):
    """ Converts a representation from 1D vector to 2D square matrix. Slightly altered from rmsd package to disregard 
    zeroes along diagonal of matrix.
    :param v: 1D input representation.
    :type v: numpy array 
    :return: Square matrix representation.
    :rtype: numpy array 
    """
    if not (np.sqrt(8 * v.shape[0] + 1) == int(np.sqrt(8 * v.shape[0] + 1))):
        print("ERROR: Can not make a square matrix.")
        exit(1)

    n = v.shape[0]
    w = ((-1 + int(np.sqrt(8 * n + 1))) // 2) + 1
    m = np.zeros((w, w))

    index = 0
    for i in range(w):
        for j in range(w):
            if i > j - 1:
                continue

            m[i, j] = v[index]
            m[j, i] = m[i, j]

            index += 1
    return m


def distance_matrix_to_coords_alt_vector(v):
    """ Converts a (2D square) distance matrix representation of a structure to Cartesian coordinates
    (first 3 columns correspond to 3D xyz coordinates) via a Gram matrix (DIFFERENT DEFINITION OF M).
    Should work exactly the same as distance_matrix_to_coords.
    :param v: 1D vector, numpy array
    :return: 3D Cartesian coordinates, numpy array
    """

    d = vector_to_matrix(v)

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


def pca_dr(n_dim, matrix):
    """
    Standardizes and does PCA on input matrix with specified number of dimensions. Outputs information used to later
    generate xyz files in the reduced dimensional space and also for the function that filters out distances between
    key atoms and their neighbors.
    :param n_dim: int
    :param matrix: array
    :return:
    """

    # Standardizing the data using StandardScaler
    pca_pipeline = pipeline.make_pipeline(preprocessing.StandardScaler(), decomposition.PCA(n_components=n_dim))
    pcafull_pipeline = pipeline.make_pipeline(preprocessing.StandardScaler(), decomposition.PCA())

    # Unscaled values for comparison
    # unscaled_clf = pipeline.make_pipeline(decomposition.PCA(n_components=n_dim, svd_solver='full'))
    # pca = unscaled_clf.named_steps['pca']

    pca_std = pca_pipeline.named_steps['pca']
    pcafull = pcafull_pipeline.named_steps['pca']

    # pca_std.fit(matrix)
    matrix_pca_fit = pca_std.fit(pd.DataFrame(matrix))
    # unscaled_matrix_pca_fit = pca.fit(pd.DataFrame(matrix))
    matrix_pca = pca_std.transform(pd.DataFrame(matrix))

    pcafull.fit(pd.DataFrame(matrix))

    # Show first principal components
    # print('\nPC 1 without scaling:\n', sum(pca.components_[0]))
    # print('\nPC 1 with scaling:\n', sum(pca_std.components_[0]))

    x_1_2_3 = []
    for i in range(0, n_dim):
        xi = np.dot(matrix_pca[:, i, None], pca_std.components_[None, i, :]) + pca_std.mean_
        x_1_2_3.append(xi)

    x_all = np.dot(matrix_pca, pca_std.components_) + pca_std.mean_

    x_1_2_3 = np.array(x_1_2_3)
    x_all = np.array(x_all)

    return matrix_pca, matrix_pca_fit, pca_std.components_, pca_std.mean_, pcafull.explained_variance_, x_1_2_3, x_all


def calc_ij(k, n):
    """
    Calculate indexes i and j of a square symmetric matrix given upper triangle vector index k and matrix side length n.
    :param k: vector index
    :param n: side length of resultant matrix M
    :return: i, j as ints
    """
    i = n - 2 - math.floor((np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2) - 0.5)
    j = k + i + 1 - (n * (n - 1) / 2) + ((n - i) * ((n - i) - 1) / 2)
    return int(i), int(j)


def top_values(a, n):
    """
    Determine indexes of n top values of matrix a
    :param a: matrix
    :param n: integer, number of top values desired
    :return: sorted list of indexes of n top values of a
    """
    return np.argsort(a)[::-1][:n]


def kabsch(coords):
    """Kabsch algorithm to get orientation of axes that minimizes RMSD (to avoid rotations in visualization). All
    structures will be aligned to the first structure in the trajectory. Only necessary when input_type="Distances",
    because structures will be generated in reduced dimensional space in an arbitrary rotational configuration.
    :param coords: coordinates along trajectory to be aligned, list or array
    """
    coords = np.array(coords)
    coords[0] -= rmsd.centroid(coords[0])
    coords_kabsch = []
    for i in range(len(coords)):
        coords[i] -= rmsd.centroid(coords[i])
        # coords_kabschi = rmsd.kabsch_rotate(coords[i][:, np.newaxis], coords[0][:, np.newaxis])
        coords_kabschi = rmsd.kabsch_rotate(coords[i], coords[0])
        coords_kabsch.append(coords_kabschi)

    coords_kabsch = np.array(coords_kabsch)

    return coords_kabsch


def chirality_test(coords, a1, a2, a3, a4):
    """ Determines chirality of structure so it is consistent throughout the generated reduced dimensional
    IRC/trajectory.
    :param coords: xyz coordinates along IRC or trajectory
    :param a[n]: 4 atom numbers that represent groups around a chiral center
    :type coords: numpy array
    :type a[n]: integers
    """

    signs = []
    for i in range(len(coords)):
        m = np.ones((4, 4))
        m[0, 0:3] = coords[i][a1 - 1]
        m[1, 0:3] = coords[i][a2 - 1]
        m[2, 0:3] = coords[i][a3 - 1]
        m[3, 0:3] = coords[i][a4 - 1]
        if np.linalg.det(m) < 0:
            signs.append(-1)
        elif np.linalg.det(m) > 0:
            signs.append(1)
        elif np.linalg.det(m) == 0:
            signs.append(0)

    negs = np.where(np.array(signs) < 0)
    poss = np.where(np.array(signs) > 0)

    print("\nDetermining chirality of structure at each point of file...")
    print("Number of structures with negative determinant (enantiomer 1): %s" % np.size(negs))
    print("Number of structures with positive determinant (enantiomer 2): %s" % np.size(poss))
    print("Total structures: %s" % len(signs))

    return negs, poss, signs


def chirality_changes_new(coords_reconstr, a1, a2, a3, a4, signs_orig):
    """ Determines chirality of structure along original trajectory and reconstructed reduced dimensional trajectory
     and switches inconsistencies along reduced dimensional IRC/trajectory.
    :param coords_reconstr: coordinates of trajectory in the reduced dimensional space
    :param a1, a2, a3, a4: indexes of atoms surrounding stereogenic center
    :param signs_orig: signs (positive or negative) that represent chirality at given point along original trajectory,
    numpy array
    """

    pos, neg, signs_reconstr = chirality_test(coords_reconstr, a1, a2, a3, a4)
    coords = coords_reconstr

    for i in range(len(signs_orig)):
        if signs_reconstr[i] != signs_orig[i]:
            # Switch sign of signs_reconstr by reflecting coordinates of that point
            # print("Switching chirality of structure %s...\n" % i)

            # Three coordinates switched
            coords[i] = -coords[i]

    return coords


def make_xyz_files(name, atoms, xyz_coords):
    """ Save principal coordinates as xyz files coord[n].xyz to output directory.
    :param atoms: atoms in input trajectory, list
    :param name: name of the input system, str
    :param xyz_coords: xyz coordinates of structures along Xi, list or numpy array
    """

    for k in range(np.array(xyz_coords).shape[0]):
        f = open('%s_coord%s.xyz' % (name, k + 1), 'w')

        for i in range(len(xyz_coords[k])):

            a = xyz_coords[k][i]
            a = a.tolist()
            b = []
            for j in range(len(a)):
                a[j] = ['%.5f' % x for x in a[j]]
                a[j].insert(0, atoms[j])
                b.append(a[j])

            f.write('%d' % len(atoms) + '\n')
            f.write('%s point %i' % (name, i + 1) + '\n')
            f.write('%s' % str(np.asarray(b)).replace("[", "").replace("]", "").replace("'", "") + '\n')

        f.close()


def goodness_of_fit(w, ndim):
    """ Calculate "goodness of fit" by determining how much of the variance (sum of w's) is described by the singular
    values used in the reduced dimensional representation.
    """
    gof = np.zeros((ndim + 1, 1))
    x = 0
    for k in range(ndim):
        x = x + w[k] / np.sum(w)
        gof[k] = x

        print("Goodness of fit (%s-dim)= %f" % (k + 1, gof[k]))

    return gof


def plot_gof(w, name, directory):

    fig = plt.figure(figsize=(8, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)

    normed_w = w / np.sum(w)
    x = range(len(w))

    ax.scatter(x, normed_w, c='k')

    ax.set_xlabel("Principal Component", fontsize=16)
    ax.set_ylabel("Proportion of Variance", fontsize=16)
    ax.set_ylim(-0.1, 1.1)

    cumulative = np.cumsum(normed_w)

    ax1.scatter(x, cumulative)
    ax1.set_xlabel("Principal Component", fontsize=16)
    ax1.set_ylabel("Cumulative Prop. of Var.", fontsize=16)
    ax1.set_ylim(-0.1, 1.1)

    maintitle = "Proportion of Variance Described by Principal Components"

    fig.tight_layout()
    # fig.suptitle("%s" % maintitle, fontsize=16)
    # fig.subplots_adjust(top=0.88)
    fig.savefig(directory + "/" + '%s_proportion_of_variance.png' % name, dpi=600)
    pd.DataFrame(normed_w).to_csv(directory + "/" + name + '_singular_vals.txt', sep='\t', index=None)


def stress_calc(d, dred, ndim):
    """ Calculate "stress" by determining how far the reconstructed points are from the original points.
    """
    stress = np.sum(np.square(dred - d)) / np.sum(np.square(d))
    print("Stress (%s-dim) = %f" % (ndim, stress))

    return stress


def print_distance_coeffs_to_files(directory, n_dim, name, pca_components):

    num_atoms = int(pca_components.shape[1]/3)

    for n in range(n_dim):
        d = []
        for k in range(num_atoms):
            i, j = calc_ij(k, num_atoms)
            coeff = pca_components[k]
            d.append({'atom 1': i, 'atom 2': j, 'Coefficient of Distance': coeff})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(directory + "/" + name + '_PC%s_components.txt' % n, sep='\t', index=None)


def transform_new_data(new_traj, directory, n_dim, a1, a2, a3, a4, pca_fit, pca_components, pca_mean, old_data, lengths=None,
                       input_type="Coordinates", mass_weighting=False):
    """
    Takes as input a new trajectory (xyz file) for a given system for which dimensionality reduction has already been
    conducted and transforms this new data into the reduced dimensional space. Generates a plot, with the new data atop
    the "trained" data, and generates xyz files for the new trajectories represented by the principal components.
    :param new_traj: new trajectory (xyz file location), str
    :param directory: output directory, str
    :param n_dim: number of dimensions of the reduced dimensional space, int
    :param a1, a2, a3, a4: four atoms surrounding stereogenic center, ints
    :param pca_fit: fit from PCA on training data
    :param pca_components: components from PCA on training data, array
    :param pca_mean: mean of input data to PCA (mean structure as coords or distances), array
    :param old_data: data that has been previously plotted (e.g., training data in reduced dimensional space), list or
    array
    :param lengths: lengths of all trajectories in training data set, list (generated by dr_routine)
    :param input_type: type of input (either "Coordinates" or "Distances"), str
    """

    print("\nTransforming %s into reduced dimensional representation..." % new_traj)

    name, energies, atoms, coordinates_all = read_file(new_traj)
    coordinates_shifted = set_atom_one_to_origin(coordinates_all)

    if mass_weighting is True:
        atom_masses, mass_weighted_coords = mass_weighting_pt(atoms, coordinates_shifted)
        coords_for_analysis = mass_weighted_coords

    else:
        coords_for_analysis = coordinates_shifted

    negatives, positives, all_signs = chirality_test(coordinates_all, a1, a2, a3, a4)

    if not os.path.exists(directory):
        os.makedirs(directory)
    print("\nResults for %s input will be stored in %s" % (new_traj, directory))

    if input_type == "Coordinates":
        # Align structures using Kabsch algorithm so rotations don't affect PCs
        coords_for_analysis = kabsch(coords_for_analysis)
        coords_for_analysis = np.reshape(coords_for_analysis, (coords_for_analysis.shape[0],
                                                               coords_for_analysis.shape[1] *
                                                               coords_for_analysis.shape[2]))

    elif input_type == "Distances":
        d2 = generate_ds(coords_for_analysis)
        coords_for_analysis = reshape_ds(d2)

    components = pca_fit.transform(coords_for_analysis)
    components_df = pd.DataFrame(components)

    x_1_2_3 = []
    for i in range(0, n_dim):
        xi = np.dot(components[:, i, None], pca_components[None, i, :]) + pca_mean
        x_1_2_3.append(xi)

    x_all = np.dot(components, pca_components) + pca_mean

    x_1_2_3 = np.array(x_1_2_3)
    x_all = np.array(x_all)

    if input_type == "Coordinates":
        # Reshape n x 3N x 1 arrays into n x N x 3 arrays
        x_1_2_3 = np.reshape(x_1_2_3, (x_1_2_3.shape[0], x_1_2_3.shape[1],
                                                     int(x_1_2_3.shape[2] / 3), 3))

        x_all = np.reshape(x_all, (1, x_all.shape[0], int(x_all.shape[1] / 3), 3))

        if mass_weighting is True:
            # Remove mass-weighting of coordinates, individual Xs
            no_mass_weighting_xyz_coords = [unmass_weighting_pt(atoms, x_1_2_3[i]) for i in range(n_dim)]

            # Remove mass-weighting of coordinates, all Xs combined into one array
            no_mass_weighting_xyz_coords_x_all = unmass_weighting_pt(atoms, x_all)

        else:
            no_mass_weighting_xyz_coords = [x_1_2_3[i] for i in range(n_dim)]
            no_mass_weighting_xyz_coords_x_all = x_all

        make_xyz_files(directory + "/" + name, atoms, no_mass_weighting_xyz_coords)
        make_xyz_files(directory + "/" + name + "_all", atoms, no_mass_weighting_xyz_coords_x_all)

    elif input_type == "Distances":
        # Turning distance matrix representations of structures back into Cartesian coordinates
        coords_cartesian_x = [[distance_matrix_to_coords_alt_vector(x_1_2_3[i][k])
                               for k in range(x_1_2_3.shape[1])] for i in range(x_1_2_3.shape[0])]

        if mass_weighting is True:
            # Remove mass-weighting of coordinates, individual Xs
            no_mass_weighting_coords_cartesian_x = [unmass_weighting_pt(atoms, coords_cartesian_x[i])
                                                    for i in range(n_dim)]
        else:
            no_mass_weighting_coords_cartesian_x = [coords_cartesian_x[i] for i in range(n_dim)]

        xyz_file_coords_cartesian = \
            [kabsch(chirality_changes_new(no_mass_weighting_coords_cartesian_x[i], a1, a2, a3, a4,
                                                          all_signs)) for i in range(n_dim)]

        # Turning distance matrix representations of structures back into Cartesian coordinates (all chosen Xs combined
        # into one xyz file)
        coords_cartesian_x_all = [distance_matrix_to_coords_alt_vector(x_all[i])
                                  for i in range(np.array(x_all).shape[0])]

        if mass_weighting is True:
            # Remove mass-weighting of coordinates, all Xs combined into one array
            no_mass_weighting_coords_cartesian_all_x = unmass_weighting_pt(atoms, coords_cartesian_x_all)
        else:
            no_mass_weighting_coords_cartesian_all_x = coords_cartesian_x_all


        # Reorient coordinates so they are in a consistent coordinate system/chirality, all Xs combined into one array
        xyz_file_coords_cartesian_all_x = \
            kabsch(chirality_changes_new(no_mass_weighting_coords_cartesian_all_x, a1, a2, a3, a4,
                                                         all_signs))

        xyz_file_coords_cartesian_all_x = np.reshape(xyz_file_coords_cartesian_all_x,
                                                     (1,
                                                      xyz_file_coords_cartesian_all_x.shape[0],
                                                      xyz_file_coords_cartesian_all_x.shape[1],
                                                      xyz_file_coords_cartesian_all_x.shape[2]))

        xyz_file_coords_cartesian = np.real(xyz_file_coords_cartesian)
        xyz_file_coords_cartesian_all_x = np.real(xyz_file_coords_cartesian_all_x)

        make_xyz_files(directory + "/" + name + "_D", atoms, xyz_file_coords_cartesian)
        make_xyz_files(directory + "/" + name + "_all_D", atoms, xyz_file_coords_cartesian_all_x)

    old_data_df = pd.DataFrame(old_data)

    return components_df

    # if lengths is not None:
    #     colorplot(old_data_df[0], old_data_df[1], old_data_df[2], same_axis=False, input_type=input_type,
    #           new_data=components_df, lengths=lengths)
    # else:
    #     colorplot(old_data_df[0], old_data_df[1], old_data_df[2], same_axis=False, input_type=input_type,
    #           new_data=components_df, output_directory=directory, imgname=(name + input_type + "new_data"))


def dr_routine(dr_input, n_dim, a1=1, a2=2, a3=3, a4=4, input_type="Coordinates", mass_weighting=False,
               filtered_distances=False, n_top_atoms=50, dist_threshold=7.0, number_of_dists=10):
    """
    Workhorse function for doing dimensionality reduction on xyz files. Dimensionality reduction can be done on the
    structures represented as Cartesian coordinates (easy/faster) or the structures represented as distances matrices
    (slower, but potentally more useful for certain systems that vary in non-linear ways, e.g., torsions).
    :param dr_input: xyz file or directory filled with xyz files that will be used to generate the reduced dimensional
    space, str
    :param n_dim: number of dimensions to reduce system to using PCA, int
    :param a1, a2, a3, a4: atom indexes surrounding stereogenic center, ints
    :param input_type: input type to PCA, either "Coordinates" or "Distances, str
    :param filtered_distances: whether, after "Coordinates" input to PCA, important distances should be determined
    using the filter_top_distances method, bool
    :param n_top_atoms: number of atoms involved in principal components to define scope of filter_top_distances, int
    :param dist_threshold: distance from the n_top_atoms to consider when determining most important distances in
    filter_top_distances, float
    :param number_of_dists: number of distances to plot after running filter_top_distances, int
    :return: lengths, name, directory, coordinates_pca, coordinates_pca_fit, coordinates_components, coordinates_mean,
    coordinates_values, coordinates_all, no_mass_weighting_xyz_coords_x_all
    """

    # Make sure even large matrices are printed out in their entirety (for the generation of xyz files)
    np.set_printoptions(threshold=np.nan)

    # Check if input is directory (containing input files) or a single input file itself
    lengths = []
    if os.path.isfile(dr_input) is True:
        print("\nInput is a file!")

        name, energies, atoms, coordinates_all = read_file(dr_input)
        coordinates_shifted = set_atom_one_to_origin(coordinates_all)

        if mass_weighting is True:
            atom_masses, mass_weighted_coords = mass_weighting_pt(atoms, coordinates_shifted)
            coords_for_analysis = mass_weighted_coords

            name = name + "_MW"

        else:
            coords_for_analysis = coordinates_shifted

            name = name + "_noMW"

        print("\nTotal number of atoms: %s\n" % coordinates_all.shape[1])

        negatives, positives, all_signs = chirality_test(coordinates_all, a1, a2, a3, a4)

        # Creating a directory for output (if directory doesn't already exist)
        directory = name + "_output"
        if not os.path.exists(directory):
            os.makedirs(directory)
        print("Results for %s input will be stored in %s" % (dr_input, directory))

    elif os.path.isdir(dr_input) is True:
        print("\nInput is a directory!")
        print("\nDoing dimensionality reduction on files in %s" % dr_input)
        xyz_files = sorted(glob.glob(dr_input + "/" + "*.xyz"))

        # Subroutine for if there are multiple files to use
        names = []
        i = 0
        for xyz_file in xyz_files:
            i = i + 1
            name, energies, atoms, coordinates_all = read_file(xyz_file)
            coordinates_shifted = set_atom_one_to_origin(coordinates_all)

            if mass_weighting is True:
                atom_masses, mass_weighted_coords = mass_weighting_pt(atoms, coordinates_shifted)
                coords_for_analysis_single = mass_weighted_coords

                name = name + "_MW"

            else:
                coords_for_analysis_single = coordinates_shifted

                name = name + "_noMW"

            names.append(name)

            lengths.append(coords_for_analysis_single.shape[0])

            if i == 1:
                coords_for_analysis = coords_for_analysis_single
            else:
                coords_for_analysis = np.concatenate((coords_for_analysis, coords_for_analysis_single), axis=0)

        print("\nTotal number of atoms per file: %s" % coordinates_all.shape[1])

        negatives, positives, all_signs = chirality_test(coords_for_analysis, a1, a2, a3, a4)

        # Creating a directory for output (if directory doesn't already exist)
        directory = os.path.basename(dr_input) + "_output"
        if not os.path.exists(directory):
            os.makedirs(directory)
        print("\nResults for structures represented as %s when input to PCA will be stored in %s" % (input_type,
                                                                                                     directory))

    else:
        print("\nERROR: As the first argument in dr_routine, analyze a single file by specifying that file's name "
              "(complete path to file) or multiple files by specifying the directory in which those trajectories are "
              "held")

    if input_type == "Coordinates":

        coords_for_analysis = np.reshape(coords_for_analysis, (coords_for_analysis.shape[0],
                                                               coords_for_analysis.shape[1] *
                                                               coords_for_analysis.shape[2]))

        # PCA
        coordinates_pca, coordinates_pca_fit, coordinates_components, coordinates_mean, coordinates_values, \
        x_1_2_3_coords, x_all_coords = pca_dr(n_dim, coords_for_analysis)

        plot_gof(coordinates_values, name, directory)

        print("\n(1/4) Done with PCA of %s!" % input_type)

        # Reshape n x 3N x 1 arrays into n x N x 3 arrays
        x_1_2_3_coords = np.reshape(x_1_2_3_coords, (x_1_2_3_coords.shape[0], x_1_2_3_coords.shape[1],
                                                     int(x_1_2_3_coords.shape[2] / 3), 3))

        print("\n(2/4) Done making individual principal coordinates, X1-%s!" % n_dim)

        x_all_coords = np.reshape(x_all_coords, (1, x_all_coords.shape[0], int(x_all_coords.shape[1] / 3), 3))

        print("\n(3/4) Done making top %s combined principal coordinates, X_all!" % n_dim)

        if mass_weighting is True:
            # Remove mass-weighting of coordinates, individual Xs
            no_mass_weighting_xyz_coords = [unmass_weighting_pt(atoms, x_1_2_3_coords[i]) for i in range(n_dim)]

            # Remove mass-weighting of coordinates, all Xs combined into one array/reduced dimensional trajectory
            no_mass_weighting_xyz_coords_x_all = unmass_weighting_pt(atoms, x_all_coords)

        else:
            no_mass_weighting_xyz_coords = [x_1_2_3_coords[i] for i in range(n_dim)]
            no_mass_weighting_xyz_coords_x_all = x_all_coords

        # Make xyz files from final coordinate arrays
        if os.path.isfile(dr_input) is True:
            make_xyz_files(directory + "/" + name, atoms, no_mass_weighting_xyz_coords)
            make_xyz_files(directory + "/" + name + "_all", atoms, no_mass_weighting_xyz_coords_x_all)

        elif os.path.isdir(dr_input) is True:
            for x in range(len(lengths)):
                name = names[x]
                if x == 0:
                    start_index = 0
                    end_index = lengths[x]
                    one_file_x = np.array(no_mass_weighting_xyz_coords)[:, start_index:end_index, :, :]
                    one_file_x_all = np.array(no_mass_weighting_xyz_coords_x_all)[:, start_index:end_index, :, :]
                    make_xyz_files(directory + "/" + name, atoms, one_file_x)
                    make_xyz_files(directory + "/" + name + "_all", atoms, one_file_x_all)
                else:
                    start_index = sum(lengths[:x])
                    end_index = sum(lengths[:(x + 1)])
                    one_file_x = np.array(no_mass_weighting_xyz_coords)[:, start_index:end_index, :, :]
                    one_file_x_all = np.array(no_mass_weighting_xyz_coords_x_all)[:, start_index:end_index, :, :]
                    make_xyz_files(directory + "/" + name, atoms, one_file_x)
                    make_xyz_files(directory + "/" + name + "_all", atoms, one_file_x_all)

        print("\n(4/4) Done with making xyz files!")

        if filtered_distances is True:
            sorted_pc1, sorted_pc2, sorted_pc3 = filter_top_distances(name, directory, n_dim, coordinates_components, coordinates_all,
                                         no_mass_weighting_xyz_coords_x_all, n_top_atoms, dist_threshold, number_of_dists=number_of_dists)

        return lengths, name, directory, coordinates_pca, coordinates_pca_fit, coordinates_components, \
               coordinates_mean, coordinates_values, coordinates_all, no_mass_weighting_xyz_coords_x_all

    elif input_type == "Distances" or input_type == "Inverse Distances":

        d2 = generate_ds(coords_for_analysis)

        print("\n(1/6) Generation of distance matrices done!")

        d_re = reshape_ds(d2)

        print("\n(2/6) Reshaping upper triangle of Ds into vectors done!")

        if input_type == "Inverse Distances":
            pca_input = np.reciprocal(d_re)
            name_ext = "_INV_D"
        else:
            pca_input = d_re
            name_ext = "_D"

        # PCA on distance matrix and inverse distance matrix
        d_pca, d_pca_fit, d_components, d_mean, d_values, x_1_2_3_d, x_all_d = pca_dr(n_dim, pca_input)

        plot_gof(d_values, name + "_D", directory)

        print("\n(3/6) Done with PCA of %s!" % input_type)

        if input_type == "Inverse Distances":
            x_1_2_3_d = np.reciprocal(x_1_2_3_d)
            x_all_d = np.reciprocal(x_all_d)

        # Turning distance matrix representations of structures back into Cartesian coordinates
        coords_cartesian_x = [[distance_matrix_to_coords_alt_vector(x_1_2_3_d[i][k])
                               for k in range(x_1_2_3_d.shape[1])] for i in range(x_1_2_3_d.shape[0])]

        print("\n(4/6) Done with converting distance matrices back to coordinates (X1-3)!")

        if mass_weighting is True:
            # Remove mass-weighting of coordinates, individual Xs
            no_mass_weighting_coords_cartesian_x = [unmass_weighting_pt(atoms, coords_cartesian_x[i])
                                                    for i in range(n_dim)]
        else:
            no_mass_weighting_coords_cartesian_x = [coords_cartesian_x[i] for i in range(n_dim)]

        xyz_file_coords_cartesian = \
            [kabsch(chirality_changes_new(no_mass_weighting_coords_cartesian_x[i], a1, a2, a3, a4,
                                                          all_signs)) for i in range(n_dim)]

        # Turning distance matrix representations of structures back into Cartesian coordinates (all chosen Xs combined
        # into one xyz file)
        coords_cartesian_x_all = [distance_matrix_to_coords_alt_vector(x_all_d[i])
                                  for i in range(np.array(x_all_d).shape[0])]

        print("\n(5/6) Done with converting distance matrices back to coordinates (X_all)!")

        if mass_weighting is True:
            # Remove mass-weighting of coordinates, all Xs combined into one array
            no_mass_weighting_coords_cartesian_all_x = unmass_weighting_pt(atoms, coords_cartesian_x_all)
        else:
            no_mass_weighting_coords_cartesian_all_x = coords_cartesian_x_all

        # Reorient coordinates so they are in a consistent coordinate system/chirality, all Xs combined into one array
        xyz_file_coords_cartesian_all_x = \
            kabsch(chirality_changes_new(no_mass_weighting_coords_cartesian_all_x, a1, a2, a3, a4,
                                                         all_signs))

        xyz_file_coords_cartesian_all_x = np.reshape(xyz_file_coords_cartesian_all_x,
                                                     (1,
                                                      xyz_file_coords_cartesian_all_x.shape[0],
                                                      xyz_file_coords_cartesian_all_x.shape[1],
                                                      xyz_file_coords_cartesian_all_x.shape[2]))

        xyz_file_coords_cartesian = np.real(xyz_file_coords_cartesian)
        xyz_file_coords_cartesian_all_x = np.real(xyz_file_coords_cartesian_all_x)

        if os.path.isfile(dr_input) is True:

            # Make final structures into xyz files
            make_xyz_files(directory + "/" + name + name_ext, atoms, xyz_file_coords_cartesian)
            make_xyz_files(directory + "/" + name + "_all" + name_ext, atoms, xyz_file_coords_cartesian_all_x)

            print("\n(6/6) Done with making xyz files!")

            return lengths, name, directory, d_pca, d_pca_fit, d_components, d_mean, d_values

        elif os.path.isdir(dr_input) is True:
            for x in range(len(lengths)):
                name = names[x]
                if x == 0:
                    start_index = 0
                    end_index = lengths[x]
                    one_file_x = np.array(xyz_file_coords_cartesian)[:, start_index:end_index, :, :]
                    one_file_x_all = np.array(xyz_file_coords_cartesian_all_x)[:, start_index:end_index, :, :]
                    make_xyz_files(directory + "/" + name + "_D", atoms, one_file_x)
                    make_xyz_files(directory + "/" + name + "_all_D", atoms, one_file_x_all)
                else:
                    start_index = sum(lengths[:x])
                    end_index = sum(lengths[:(x + 1)])
                    one_file_x = np.array(xyz_file_coords_cartesian)[:, start_index:end_index, :, :]
                    one_file_x_all = np.array(xyz_file_coords_cartesian_all_x)[:, start_index:end_index, :, :]
                    make_xyz_files(directory + "/" + name + "_D", atoms, one_file_x)
                    make_xyz_files(directory + "/" + name + "_all_D", atoms, one_file_x_all)

            print("\n(6/6) Done with making xyz files!")

            # return lengths, name, directory, d_pca, d_pca_fit, d_components, d_mean, d_values

        return lengths, name, directory, d_pca, d_pca_fit, d_components, d_mean, d_values