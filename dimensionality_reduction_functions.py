#!/usr/bin/env python
# coding: utf-8

import numpy as np
import calculate_rmsd as rmsd
import pandas as pd
import math
import glob
import os
import ntpath
import plotting_functions
from periodictable import *
from matplotlib import pyplot as plt
from sklearn import *
from sympy import solve, Symbol


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def read_file(path):
    """ Reads in each file, and for each file, separates each IRC point into its own matrix of cartesian coordinates.
    coordinates_all is arranged as coordinates_all[n][N][c], where n is the IRC point, N is the atom number, and c
    is the x, y, or z coordinate.
    """
    system_name = path_leaf(path)
    print("File being read is: %s" % system_name)

    extensionless_system_name = os.path.splitext(system_name)[0]

    xyz = open(path)
    n_atoms = int(xyz.readline())
    energies = []
    atoms = []
    coordinates = []
    velocities = []

    # Each point along the IRC/traj will be stored as an entry in a 3D array called coordinates_all
    coordinates_all = []
    velocities_all = []
    for line in xyz:
        splitline = line.split()
        if len(splitline) == 4:
            atom, x, y, z = line.split()
            atoms.append(atom)
            coordinates.append([float(x), float(y), float(z)])
        elif len(splitline) == 7:
            atom, x, y, z, vx, vy, vz = line.split()
            atoms.append(atom)
            coordinates.append([float(x), float(y), float(z)])
            velocities.append([float(vx), float(vy), float(vz)])
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
        velocities_all.append(velocities)

    xyz.close()

    coordinates_all = np.asarray(coordinates_all)
    velocities_all = np.asarray(velocities_all)

    # Print ERROR if length of coordinate section doesn't match number of atoms specified at beginning of xyz file
    if len(atoms) != n_atoms:
        print("ERROR: file contains %d atoms instead of the stated number %d" % (n_atoms, len(atoms)))
        print("number of atoms in file: %d" % len(atoms))
        print("number of coordinates:   %d" % len(coordinates))

    return extensionless_system_name, atoms, coordinates_all


def read_file_df(path):
    """ Reads in each file, and for each file, separates each IRC point into its own matrix of cartesian coordinates.
    coordinates_all is arranged as coordinates_all[n][N][c], where n is the IRC point, N is the atom number, and c
    is the x, y, or z coordinate.
    """
    system_name = path_leaf(path)
    print("File being read is: %s" % system_name)

    extensionless_system_name = os.path.splitext(system_name)[0]

    data = pd.read_csv(path, header=None, delim_whitespace=True, names=['atom', 'X', 'Y', 'Z'])
    n_atoms = int(data.loc[0][0])
    n_lines_per_frame = int(n_atoms + 2)

    data_array = np.array(data)

    data_reshape = np.reshape(data_array, (int(data_array.shape[0]/n_lines_per_frame), n_lines_per_frame, data_array.shape[1]))
    cartesians = data_reshape[:, 2::, 1::].astype(np.float)
    atom_list = data_reshape[0, 2::, 0]

    return extensionless_system_name, atom_list, cartesians


def set_atom_one_to_origin(coordinates):
    coordinates_shifted = coordinates - coordinates[:, np.newaxis, 0]

    return coordinates_shifted


def mass_weighting(atoms, coordinates):

    coordinates = np.array(coordinates)
    atoms = np.array(atoms)

    atom_masses = [formula(atom).mass for atom in atoms]
    weighting = np.sqrt(atom_masses)
    mass_weighted_coords = coordinates * weighting[np.newaxis, :, np.newaxis]

    return mass_weighted_coords


def remove_mass_weighting(atoms, coordinates):

    coordinates = np.array(coordinates)
    atoms = np.array(atoms)

    atom_masses = [formula(atom).mass for atom in atoms]
    weighting = np.sqrt(atom_masses)
    unmass_weighted_coords = coordinates / weighting[np.newaxis, :, np.newaxis]

    return unmass_weighted_coords


def generate_distance_matrices(coordinates):
    """ Generates distance matrices for each structure.
    """
    coordinates = np.array(coordinates)
    d2 = np.sum((coordinates[:, :, None] - coordinates[:, None, :]) ** 2, axis=3)
    return d2


def generate_dihedral_matrices(coordinates):
    return coordinates


def generate_and_reshape_ds_big_structures(coordinates):
    """ Generates matrix of pairwise distances, which includes pairwise distances for each structure. To be fed directly
    to PCA.
    :param coordinates:
    """
    coordinates = np.array(coordinates)
    atoms = int(coordinates.shape[1])
    d_re = np.zeros((coordinates.shape[0], int(atoms*(atoms-1)/2)))
    for i in range(coordinates.shape[0]):
        d2 = np.square(metrics.pairwise.euclidean_distances(coordinates[i]))
        x = d2[0].shape[0]
        dint_re = d2[np.triu_indices(x, k=1)]
        d_re[i] = dint_re
    return d_re


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


def distance_matrix_to_coords(v):
    """ Converts a (2D square) distance matrix representation of a structure to Cartesian coordinates (first 3 columns
    correspond to 3D xyz coordinates) via a Gram matrix.
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
    Does PCA on input matrix with specified number of dimensions. Outputs information used to later generate xyz files
    in the reduced dimensional space and also for the function that filters out distances between key atoms and their
    neighbors.
    :param n_dim: int
    :param matrix: array
    :return:
    """

    # If using interatomic distances, this matrix size corresponds to 500 atoms. If this is the case, incremental PCA
    # is used.
    # if matrix.shape[1] > 124750:
    #
    #     # print("Doing PCA on %s matrix" % matrix.shape)
    #
    #     # print("Large matrix. Doing incremental PCA...")
    #     print("Large matrix. Doing Sparse PCA...")
    #
    #     # pca = decomposition.IncrementalPCA(n_components=n_dim)
    #     pca = decomposition.SparsePCA(n_components=n_dim, tol=1e-4, verbose=100, alpha=1000)
    #     pca_full = decomposition.IncrementalPCA()
    #
    #     matrix_pca_fit = pca.fit(pd.DataFrame(matrix))
    #     matrix_pca = pca.transform(pd.DataFrame(matrix))
    #
    #     pca_full.fit(pd.DataFrame(matrix))
    #
    # else:

    unscaled_pca_pipeline = pipeline.make_pipeline(decomposition.PCA(n_components=n_dim, svd_solver='full'))
    unscaled_pca_full_pipeline = pipeline.make_pipeline(decomposition.PCA(svd_solver='full'))

    pca = unscaled_pca_pipeline.named_steps['pca']
    pca_full = unscaled_pca_full_pipeline.named_steps['pca']

    matrix_pca_fit = pca.fit(pd.DataFrame(matrix))
    matrix_pca = pca.transform(pd.DataFrame(matrix))

    pca_full.fit(pd.DataFrame(matrix))

    return matrix_pca, matrix_pca_fit, pca.components_, pca.mean_, pca_full.explained_variance_


def filter_important_distances(upper_tri_d2_matrices, num_dists=500):

    num_points = upper_tri_d2_matrices.shape[0]
    vec_length = upper_tri_d2_matrices.shape[1]

    num_atoms = calc_num_atoms(vec_length)

    variances = []
    atom_indexes = {}
    for k in range(vec_length):
        variances.append(np.var(upper_tri_d2_matrices[:, k]))
        atom1, atom2 = calc_ij(k, num_atoms)
        atom_indexes[k] = atom1, atom2

    important_distances_matrix = np.zeros((num_points, num_dists))
    top_vars_indexes = top_values_indexes(variances, num_dists)

    i = 0
    selected_dist_atom_indexes = {}
    for index in top_vars_indexes:
        important_distances_matrix[:, i] = upper_tri_d2_matrices[:, index]
        selected_dist_atom_indexes[i] = atom_indexes[index]
        i += 1

    return important_distances_matrix, selected_dist_atom_indexes


def calc_num_atoms(vec_length):

    n = Symbol('n', positive=True)
    answers = solve(n * (n - 1) / 2 - vec_length, n)
    num_atoms = int(answers[0])

    return num_atoms


def generate_PC_matrices(n_dim, matrix_reduced, components, mean):

    PCs_separate = []
    for i in range(0, n_dim):
        PCi = np.dot(matrix_reduced[:, i, None], components[None, i, :]) + mean
        PCs_separate.append(PCi)

    PCs_combined = np.dot(matrix_reduced, components) + mean

    PCs_separate = np.array(PCs_separate)
    PCs_combined = np.array(PCs_combined)

    return PCs_separate, PCs_combined


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


def top_values_indexes(a, n):
    """
    Determine indexes of n top values of matrix a
    :param a: matrix
    :param n: integer, number of top values desired
    :return: sorted list of indexes of n top values of a
    """
    return np.argsort(a)[::-1][:n]


def kabsch(coords):
    """Kabsch algorithm to get orientation of axes that minimizes RMSD. All structures will be aligned to the first
    structure in the trajectory.
    :param coords: coordinates along trajectory to be aligned, list or array
    """
    coords = np.array(coords)
    coords[0] -= rmsd.centroid(coords[0])
    coords_kabsch = []
    for i in range(len(coords)):
        coords[i] -= rmsd.centroid(coords[i])
        coords_kabschi = rmsd.kabsch_rotate(coords[i], coords[0])
        coords_kabsch.append(coords_kabschi)

    return np.array(coords_kabsch)


def align_to_original_traj(coords, original_traj_coords):
    """Kabsch algorithm to get orientation of axes that minimizes RMSD (to avoid rotations in visualization). All
    structures will be aligned to the first structure in the original trajectory.
    :param coords: coordinates along trajectory to be aligned, list or array
    :param original_traj_coords: coordinates along original trajectory
    """
    coords = np.array(coords)
    coords_aligned = []
    original_traj_coords[0] -= rmsd.centroid(original_traj_coords[0])
    for i in range(len(coords)):
        coords[i] -= rmsd.centroid(coords[i])
        coords_i = rmsd.kabsch_rotate(coords[i], original_traj_coords[0])
        coords_aligned.append(coords_i)

    return np.array(coords_aligned)


def chirality_test(coords, stereo_atoms):
    """ Determines chirality of structure so it is consistent throughout the generated reduced dimensional
    IRC/trajectory.
    :param coords: xyz coordinates along IRC or trajectory
    :param stereo_atoms: list of 4 atom numbers that represent groups around a chiral center
    :type coords: numpy array
    :type stereo_atoms: list
    """
    a1 = stereo_atoms[0]
    a2 = stereo_atoms[1]
    a3 = stereo_atoms[2]
    a4 = stereo_atoms[3]
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
    zeros = np.where(np.array(signs) == 0)

    # Print statements for debugging purposes
    # print("\nDetermining chirality of structure at each point of file...")
    # print("Number of structures with negative determinant (enantiomer 1): %s" % np.size(negs))
    # print("Number of structures with positive determinant (enantiomer 2): %s" % np.size(poss))
    # print("Number of structures with zero-valued determinant (all stereo_atoms in same plane): %s" % np.size(zeros))
    # print("Total structures: %s" % len(signs))

    return negs, poss, zeros, signs


def chirality_changes_new(coords_reconstr, stereo_atoms, signs_orig):
    """ Determines chirality of structure along original trajectory and reconstructed reduced dimensional trajectory
     and switches inconsistencies along reduced dimensional IRC/trajectory.
    :param coords_reconstr: coordinates of trajectory in the reduced dimensional space
    :param stereo_atoms: list of 4 indexes of atoms surrounding stereogenic center
    :param signs_orig: signs (positive or negative) that represent chirality at given point along original trajectory,
    numpy array
    """

    pos, neg, zero, signs_reconstr = chirality_test(coords_reconstr, stereo_atoms)
    coords = coords_reconstr

    for i in range(len(signs_orig)):
        if signs_orig[i] == 0:
            # If molecule begins planar but reconstruction of PCs are not, keep chirality consistent thru PC
            if i > 0 and signs_reconstr[i] != signs_reconstr[0]:
                coords[i] = -coords[i]
        elif signs_reconstr[i] != signs_orig[i]:
            coords[i] = -coords[i]

    return coords


def make_pc_xyz_files(output_directory, title, atoms, coordinates):
    """ Save principal coordinates as xyz files PC[n].xyz to output directory.
    :param output_directory: output directory to store xyz files, str
    :param atoms: atoms in input trajectory, list
    :param title: name of the input system, str
    :param coordinates: xyz coordinates of structures along PCi, list or numpy array
    """

    for k in range(np.array(coordinates).shape[0]):
        if np.array(coordinates).shape[0] == 1:
            f = open(os.path.join(output_directory, '%s_all_PCs.xyz' % title), 'w')
        else:
            f = open(os.path.join(output_directory, '%s_PC%s.xyz' % (title, k + 1)), 'w')

        for i in range(len(coordinates[k])):

            a = coordinates[k][i]
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


def goodness_of_fit(values, ndim):
    """ Calculate "goodness of fit" by determining how much of the variance (sum of w's) is described by the singular
    values used in the reduced dimensional representation.
    """
    gof = np.zeros((ndim + 1, 1))
    x = 0
    for k in range(ndim):
        x = x + values[k] / np.sum(values)
        gof[k] = x

        print("Goodness of fit (%s-dim)= %f" % (k + 1, gof[k]))

    return gof


def plot_gof(values, name, directory):

    fig = plt.figure(figsize=(8, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)

    normed_w = values / np.sum(values)
    x = range(len(values))

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


def print_distance_coeffs_to_files(directory, n_dim, name, pca_components, num_atoms):

    for n in range(n_dim):
        d = []
        for k in range(len(pca_components[n])):
            i, j = calc_ij(k, num_atoms)
            coeff = pca_components[n][k]
            d.append({'atom 1': i, 'atom 2': j, 'Coefficient of Distance': coeff})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(directory + "/" + name + '_PC%s_components.txt' % (n+1), sep='\t', index=None)


def print_distance_coeffs_to_files_filtered(atom_indexes, n_dim, pca_components, name, directory):

    for n in range(n_dim):
        d = []
        for k in range(len(pca_components[n])):
            coeff = pca_components[n][k]
            d.append({'atom 1': atom_indexes[k][0], 'atom 2': atom_indexes[k][1], 'Coefficient of Distance': coeff})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(directory + "/" + name + '_PC%s_components.txt' % (n+1), sep='\t', index=None)


def print_distance_coeffs_to_files_weighted(directory, n_dim, name, pca_components, pca_values, num_atoms, display=False):

    for n in range(n_dim):
        d = []
        for k in range(len(pca_components[n])):
            i, j = calc_ij(k, num_atoms)
            coeff = (pca_values[n]/sum(pca_values))*pca_components[n][k]
            d.append({'atom 1': i, 'atom 2': j, 'Coefficient of Distance': coeff})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(directory + "/" + name + '_PC%s_components_weighted.txt' % (n+1), sep='\t', index=None)

        if display:
            print("PC%s" % (n+1))
            print(sorted_d)


def transform_new_data(new_input, output_directory, n_dim, pca_fit, pca_components, pca_mean, original_traj_coords,
                       stereo_atoms=[1, 2, 3, 4], input_type="Cartesians", MW=False):
    """
    Takes as input a new trajectory (xyz file) for a given system for which dimensionality reduction has already been
    conducted and transforms this new data into the reduced dimensional space. Generates a plot, with the new data atop
    the "trained" data, and generates xyz files for the new trajectories represented by the principal components.
    :param new_input: new input to dimensionality reduction (xyz file location), str
    :param output_directory: output directory, str
    :param n_dim: number of dimensions of the reduced dimensional space, int
    :param pca_fit: fit from PCA on training data
    :param pca_components: components from PCA on training data, array
    :param pca_mean: mean of input data to PCA (mean structure as coords or distances), array
    :param original_traj_coords: coordinates of the trajectory that the reduced dimensional space was trained on
    :param stereo_atoms: indexes of 4 atoms surrounding stereogenic center, list of ints
    :param input_type: type of input (either "Cartesians" or "Distances"), str
    :param MW: whether coordinates should be mass weighted prior to PCA, bool
    """

    print("\nTransforming %s into reduced dimensional representation..." % new_input)

    name, atoms, coordinates = read_file(new_input)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("\nResults for %s input will be stored in %s" % (new_input, output_directory))

    # Determining names of output directories/files
    if input_type == "Cartesians":
        file_name_end = "_Cartesians"
    elif input_type == "Distances":
        file_name_end = "_Distances"
    if MW is True:
        file_name_end = file_name_end + "_MW"
    elif MW is False:
        file_name_end = file_name_end + "_noMW"

    if input_type == "Cartesians":
        # Align structures using Kabsch algorithm so rotations don't affect PCs
        aligned_original_traj_coords = kabsch(original_traj_coords)
        coords_for_analysis = align_to_original_traj(coordinates, aligned_original_traj_coords)
        # coords_for_analysis = kabsch(coordinates)
        if MW is True:
            atom_masses, mass_weighted_coords = mass_weighting(atoms, coords_for_analysis)
            coords_for_analysis = mass_weighted_coords

        else:
            coords_for_analysis = coords_for_analysis

        coords_for_analysis = np.reshape(coords_for_analysis, (coords_for_analysis.shape[0],
                                                               coords_for_analysis.shape[1] *
                                                               coords_for_analysis.shape[2]))

    elif input_type == "Distances":

        if MW is True:
            coordinates_shifted = set_atom_one_to_origin(coordinates)
            atom_masses, mass_weighted_coords = mass_weighting(atoms, coordinates_shifted)
            coords_for_analysis = mass_weighted_coords

        else:
            coords_for_analysis = coordinates

        negatives, positives, zeroes, all_signs = chirality_test(coordinates, stereo_atoms)
        d2 = generate_distance_matrices(coords_for_analysis)
        coords_for_analysis = reshape_ds(d2)

    components = pca_fit.transform(coords_for_analysis)
    components_df = pd.DataFrame(components)

    PCs_separate = []
    for i in range(0, n_dim):
        PCi = np.dot(components[:, i, None], pca_components[None, i, :]) + pca_mean
        PCs_separate.append(PCi)

    PCs_combined = np.dot(components, pca_components) + pca_mean

    PCs_separate = np.array(PCs_separate)
    PCs_combined = np.array(PCs_combined)

    if input_type == "Cartesians":
        # Reshape n x 3N x 1 arrays into n x N x 3 arrays
        PCs_separate = np.reshape(PCs_separate, (PCs_separate.shape[0], PCs_separate.shape[1],
                                                     int(PCs_separate.shape[2] / 3), 3))

        PCs_combined = np.reshape(PCs_combined, (1, PCs_combined.shape[0], int(PCs_combined.shape[1] / 3), 3))

        if MW is True:
            # Remove mass-weighting of coordinates
            no_mass_weighting_PCs_separate = [unmass_weighting(atoms, PCs_separate[i])
                                                      for i in range(n_dim)]
            no_mass_weighting_PCs_combined = unmass_weighting(atoms, PCs_combined)
        else:
            no_mass_weighting_PCs_separate = PCs_separate
            no_mass_weighting_PCs_combined = PCs_combined

        aligned_PCs_separate = no_mass_weighting_PCs_separate
        aligned_PCs_combined = no_mass_weighting_PCs_combined

    elif input_type == "Distances":
        # Turning distance matrix representations of structures back into Cartesian coordinates
        PCs_separate = [[distance_matrix_to_coords(PCs_separate[i][k])
                               for k in range(PCs_separate.shape[1])] for i in range(PCs_separate.shape[0])]
        PCs_combined = [distance_matrix_to_coords(PCs_combined[i])
                                  for i in range(np.array(PCs_combined).shape[0])]

        PCs_separate = np.real(PCs_separate)
        PCs_combined = np.real(PCs_combined)

        if MW is True:
            # Remove mass-weighting of coordinates
            no_mass_weighting_PCs_separate = [unmass_weighting(atoms, PCs_separate[i])
                                                      for i in range(n_dim)]
            no_mass_weighting_PCs_combined = unmass_weighting(atoms, PCs_combined)
        else:
            no_mass_weighting_PCs_separate = PCs_separate
            no_mass_weighting_PCs_combined = PCs_combined

        # Reorient coordinates so they are in a consistent orientation
        aligned_PCs_separate = [kabsch(chirality_changes_new(no_mass_weighting_PCs_separate[i], stereo_atoms,
                                          all_signs)) for i in range(n_dim)]
        aligned_PCs_combined = kabsch(chirality_changes_new(no_mass_weighting_PCs_combined, stereo_atoms, all_signs))
        aligned_PCs_combined = np.reshape(aligned_PCs_combined, (1, aligned_PCs_combined.shape[0],
                                                      aligned_PCs_combined.shape[1],
                                                      aligned_PCs_combined.shape[2]))


    make_xyz_files(output_directory + "/" + name + file_name_end, atoms, aligned_PCs_separate)
    make_xyz_files(output_directory + "/" + name + file_name_end, atoms, aligned_PCs_combined)

    return components_df


def pathreducer(xyz_file_path, n_dim, stereo_atoms=[1, 2, 3, 4], input_type="Cartesians", MW=False,
                plot_variance=True, print_distance_coefficients=True, reconstruct=True):
    """
    Workhorse function for doing dimensionality reduction on xyz files. Dimensionality reduction can be done on the
    structures represented as Cartesian coordinates (easy/faster) or the structures represented as distances matrices
    (slower, but potentially more useful for certain systems that vary in non-linear ways, e.g., torsions).
    :param xyz_file_path: xyz file or directory filled with xyz files that will be used to generate the reduced dimensional
    space, str
    :param n_dim: number of dimensions to reduce system to using PCA, int
    :param stereo_atoms: list of 4 atom indexes surrounding stereogenic center, ints
    :param input_type: input type to PCA, either "Cartesians" or "Distances", str
    :return: name, directory, pca, pca_fit, components, mean, values, lengths
    """

    # Make sure even large matrices are printed out in their entirety (for the generation of xyz files)
    np.set_printoptions(threshold=np.nan)

    # Check if input is directory (containing input files) or a single input file itself
    assert os.path.isfile(xyz_file_path) or os.path.isdir(xyz_file_path), "No such file or directory."

    # Determining names of output directories/files
    if input_type == "Cartesians":
        file_name_end = "_Cartesians"
    elif input_type == "Distances":
        file_name_end = "_Distances"
    if MW is True:
        file_name_end = file_name_end + "_MW"
    elif MW is False:
        file_name_end = file_name_end + "_noMW"

    file_lengths = []
    if os.path.isfile(xyz_file_path) is True:
        print("\nInput is one file.")
        name, atoms, coordinates = read_file(xyz_file_path)
        coords_for_analysis = coordinates

    elif os.path.isdir(xyz_file_path) is True:
        print("\nInput is a directory of files.")
        print("\nDoing dimensionality reduction on files in %s" % xyz_file_path)

        # TODO: FIX TO BE ABLE TO DEAL WITH WINDOWS GLOB BEHAVIOR
        xyz_files = sorted(glob.glob(xyz_file_path + "/" + "*.xyz"))

        # Subroutine for if the input specified is a directory of xyz files
        names = []
        atoms = []
        i = 0
        for xyz_file in xyz_files:
            i = i + 1
            name, atoms_one_file, coordinates = read_file(xyz_file)
            names.append(name)
            atoms.append(atoms_one_file)
            file_lengths.append(coordinates.shape[0])

            if i == 1:
                coords_for_analysis = coordinates
            else:
                coords_for_analysis = np.concatenate((coords_for_analysis, coordinates), axis=0)

        name = "multiple_files"

    else:
        print("\nERROR: As the first argument in pathreducer, analyze a single file by specifying that file's name "
              "(complete path to file) or multiple files by specifying the directory in which those trajectories are "
              "held")

    # Creating a directory for output (if directory doesn't already exist)
    output_directory = name + file_name_end + "_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Results for %s input will be stored in %s" % (xyz_file_path, output_directory))

    if input_type == "Cartesians":

        coords_for_PCA = kabsch(coords_for_analysis)
        print("\n(1C) Done aligning structures using Kabsch algorithm")

        if MW is True:
            atom_masses, mass_weighted_coordinates = mass_weighting(atoms, coords_for_PCA)
            coords_for_PCA = mass_weighted_coordinates

            print("\n(MW) Done mass-weighting coordinates!")

        coords_for_PCA = np.reshape(coords_for_PCA, (coords_for_PCA.shape[0],
                                                               coords_for_PCA.shape[1] *
                                                               coords_for_PCA.shape[2]))

        # PCA
        cartesians_pca, cartesians_pca_fit, cartesians_components, cartesians_mean, cartesians_values = \
            pca_dr(n_dim, coords_for_PCA)
        PCs_separate, PCs_combined = generate_PC_matrices(n_dim, cartesians_pca, cartesians_components, cartesians_mean)

        if plot_variance:
            plot_gof(cartesians_values, name + file_name_end, output_directory)

        print("\n(2) Done with PCA of %s!" % input_type)
        print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")

        if reconstruct:
            # Reshape n x 3N x 1 arrays into n x N x 3 arrays
            PCs_separate = np.reshape(PCs_separate, (PCs_separate.shape[0], PCs_separate.shape[1],
                                                         int(PCs_separate.shape[2] / 3), 3))

            PCs_combined = np.reshape(PCs_combined, (1, PCs_combined.shape[0], int(PCs_combined.shape[1] / 3), 3))

            if MW is True:
                # Remove mass-weighting of coordinates, individual Xs
                no_mass_weighting_PCs_separate = [remove_mass_weighting(atoms, PCs_separate[i]) for i in range(n_dim)]

                # Remove mass-weighting of coordinates, all Xs combined into one array/reduced dimensional trajectory
                no_mass_weighting_PCs_combined = remove_mass_weighting(atoms, PCs_combined)

                print("\n(UMW) Done removing mass-weighting!")

            else:
                no_mass_weighting_PCs_separate = [PCs_separate[i] for i in range(n_dim)]
                no_mass_weighting_PCs_combined = PCs_combined

            # Make xyz files from final coordinate arrays
            if os.path.isfile(xyz_file_path) is True:
                make_xyz_files(output_directory + "/" + name + file_name_end, atoms, no_mass_weighting_PCs_separate)
                make_xyz_files(output_directory + "/" + name + file_name_end, atoms, no_mass_weighting_PCs_combined)

                print("\n(4) Done with making output xyz files!")

            elif os.path.isdir(xyz_file_path) is True:
                for x in range(len(file_lengths)):
                    filename = names[x]
                    if x == 0:
                        start_index = 0
                        end_index = file_lengths[x]
                        one_file_PCs_separate = np.array(no_mass_weighting_PCs_separate)[:, start_index:end_index, :, :]
                        one_file_PCs_combined = np.array(no_mass_weighting_PCs_combined)[:, start_index:end_index, :, :]
                        make_xyz_files(output_directory + "/" + filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                        make_xyz_files(output_directory + "/" + filename + file_name_end, atoms_one_file, one_file_PCs_combined)
                    else:
                        start_index = sum(file_lengths[:x])
                        end_index = sum(file_lengths[:(x + 1)])
                        one_file_PCs_separate = np.array(no_mass_weighting_PCs_separate)[:, start_index:end_index, :, :]
                        one_file_PCs_combined = np.array(no_mass_weighting_PCs_combined)[:, start_index:end_index, :, :]
                        make_xyz_files(output_directory + "/" + filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                        make_xyz_files(output_directory + "/" + filename + file_name_end, atoms_one_file, one_file_PCs_combined)

            print("\nDone generating output!")

        return name, output_directory, cartesians_pca, cartesians_pca_fit, cartesians_components, \
               cartesians_mean, cartesians_values, file_lengths, coords_for_analysis

    elif input_type == "Distances":

        aligned_coordinates = kabsch(coordinates)

        if MW is True:
            coordinates_shifted = set_atom_one_to_origin(coordinates)
            atom_masses, mass_weighted_coordinates = mass_weighting(atoms, coordinates_shifted)
            coords_for_PCA = mass_weighted_coordinates

            print("\n(MW) Done mass-weighting coordinates!")

        else:
            coords_for_PCA = coords_for_analysis

        negatives, positives, zeroes, all_signs = chirality_test(coords_for_analysis, stereo_atoms)

        if coords_for_PCA.shape[1] > 1000:
            num_dists = 70000
            print("Big matrix. Using the top %s distances for PCA..." % num_dists)
            d2_re_matrix = generate_and_reshape_ds_big_structures(coords_for_PCA)
            d_re, selected_dist_atom_indexes = filter_important_distances(d2_re_matrix, num_dists=num_dists)
            # TODO: Make reconstruction possible by setting weights on all "non-important" distances to zero
            reconstruct = False
        else:
            d2 = generate_distance_matrices(coords_for_PCA)
            d_re = reshape_ds(d2)

        print("\n(1D) Generation of distance matrices and reshaping upper triangles into vectors done!")

        # PCA on distance matrix
        d_pca, d_pca_fit, d_components, d_mean, d_values = pca_dr(n_dim, d_re)
        PCs_separate_d, PCs_combined_d = generate_PC_matrices(n_dim, d_pca, d_components, d_mean)

        if plot_variance:
            plot_gof(d_values, name + file_name_end, output_directory)

        if print_distance_coefficients:
            if coords_for_PCA.shape[1] > 1000:
                print_distance_coeffs_to_files_filtered(selected_dist_atom_indexes, n_dim, d_components,
                                                        name + file_name_end, output_directory)
            else:
                print_distance_coeffs_to_files(output_directory, n_dim, name + file_name_end, d_components, len(atoms))

        print("\n(2) Done with PCA of %s!" % input_type)
        print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")

        if reconstruct:
            # Turning distance matrix representations of structures back into Cartesian coordinates
            PCs_separate = [[distance_matrix_to_coords(PCs_separate_d[i][k])
                                   for k in range(PCs_separate_d.shape[1])] for i in range(PCs_separate_d.shape[0])]
            # Turning distance matrix representations of structures back into Cartesian coordinates (all chosen Xs combined
            # into one xyz file)
            PCs_combined = [distance_matrix_to_coords(PCs_combined_d[i])
                                      for i in range(np.array(PCs_combined_d).shape[0])]

            PCs_separate = np.real(PCs_separate)
            PCs_combined = np.real(PCs_combined)

            print("\n(4D)-(6D) Done with converting distance matrices back to Cartesian coordinates!")

            if MW is True:
                # Remove mass-weighting of coordinates, individual PCs
                no_mass_weighting_PCs_separate = [unmass_weighting(atoms, PCs_separate[i])
                                                  for i in range(n_dim)]
                no_mass_weighting_PCs_combined = unmass_weighting(atoms, PCs_combined)
                print("\n(UMW) Done removing mass-weighting!")

            else:
                no_mass_weighting_PCs_separate = PCs_separate
                no_mass_weighting_PCs_combined = PCs_combined

            chirality_consistent_PCs_separate = [chirality_changes_new(no_mass_weighting_PCs_separate[i], stereo_atoms, all_signs)
                                                                   for i in range(n_dim)]

            # Reorient coordinates so they are in a consistent coordinate system/chirality, all Xs combined into one array
            chirality_consistent_PCs_combined = kabsch(chirality_changes_new(no_mass_weighting_PCs_combined, stereo_atoms, all_signs))

            chirality_consistent_PCs_combined = np.reshape(chirality_consistent_PCs_combined,
                                                         (1,
                                                          chirality_consistent_PCs_combined.shape[0],
                                                          chirality_consistent_PCs_combined.shape[1],
                                                          chirality_consistent_PCs_combined.shape[2]))

            # Align new Cartesian coordinates to ALIGNED original trajectory
            aligned_PCs_separate = [align_to_original_traj(chirality_consistent_PCs_separate[i], aligned_coordinates)
                                                      for i in range(len(chirality_consistent_PCs_separate))]
            aligned_PCs_combined = [align_to_original_traj(chirality_consistent_PCs_combined[i], aligned_coordinates)
                                                      for i in range(len(chirality_consistent_PCs_combined))]

            print("\n(7D) Done checking chirality of resultant structures!")
            print("\n(8D) Done aligning!")

            if os.path.isfile(xyz_file_path) is True:

                # Make final structures into xyz files
                make_xyz_files(output_directory + "/" + name + file_name_end, atoms, aligned_PCs_separate)
                make_xyz_files(output_directory + "/" + name + file_name_end, atoms, aligned_PCs_combined)

            elif os.path.isdir(xyz_file_path) is True:
                for x in range(len(file_lengths)):
                    filename = names[x]
                    if x == 0:
                        start_index = 0
                        end_index = file_lengths[x]
                        one_file_PCs_separate = np.array(aligned_PCs_separate)[:, start_index:end_index, :, :]
                        one_file_PCs_combined = np.array(aligned_PCs_combined)[:, start_index:end_index, :, :]
                        make_xyz_files(output_directory + "/" + filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                        make_xyz_files(output_directory + "/" + filename + file_name_end, atoms_one_file, one_file_PCs_combined)
                    else:
                        start_index = sum(file_lengths[:x])
                        end_index = sum(file_lengths[:(x + 1)])
                        one_file_PCs_separate = np.array(aligned_PCs_separate)[:, start_index:end_index, :, :]
                        one_file_PCs_combined = np.array(aligned_PCs_combined)[:, start_index:end_index, :, :]
                        make_xyz_files(output_directory + "/" + filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                        make_xyz_files(output_directory + "/" + filename + file_name_end, atoms_one_file, one_file_PCs_combined)

        print("\nDone generating output!")

        return name, output_directory, d_pca, d_pca_fit, d_components, d_mean, d_values, file_lengths, aligned_coordinates


def generate_deformation_vector(start_structure, end_structure):

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    nmd_coords = np.reshape(start_structure, (1, np.array(start_structure).shape[0]*np.array(start_structure).shape[1]))

    deformation_vector = end_structure - start_structure
    deformation_vector = np.reshape(deformation_vector,
                                    (1, np.array(deformation_vector).shape[0]*np.array(deformation_vector).shape[1]))
    print("NMD Coordinates:", nmd_coords)
    print("Deformation vector:", deformation_vector)

    return deformation_vector
