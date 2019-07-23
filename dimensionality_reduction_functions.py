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


def read_file_df(path):
    """ Reads in an xyz file from path as a DataFrame. This DataFrame is then turned into a 3D array such that the
    dimensions are (number of points) X (number of atoms) X 3 (Cartesian coordinates). The system name (based on the
    filename), list of atoms in the system, and Cartesian coordinates are output.
    :param path: path to xyz file to be read
    :return extensionless_system_name: str
            atom_list: numpy array
            cartesians: numpy array
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


def pca_dr(matrix):
    """
    Does PCA on input matrix with specified number of dimensions. Outputs information used to later generate xyz files
    in the reduced dimensional space and also for the function that filters out distances between key atoms and their
    neighbors.
    :param matrix: array
    :return: matrix_pca: input data (in matrix) projected onto covariance matrix eigenvectors (PCs)
            matrix_pca_fit: fit used to transform new data into reduced dimensional space
            covar_matrix.components_: eigenvectors of covariance matrix
            covar_matrix.mean_: mean of the original dataset (in matrix)
            covar_matrix.explained_variance_: amount of variance described by each PC
    """

    matrix = pd.DataFrame(matrix)

    covar_matrix = decomposition.PCA()

    matrix_pca_fit = covar_matrix.fit(matrix)
    matrix_pca = covar_matrix.transform(matrix)

    return matrix_pca, matrix_pca_fit, covar_matrix.components_, covar_matrix.mean_, covar_matrix.explained_variance_

#TODO: Add function that is able to do LDA on data rather than PCA
def lca_dr(matrix, data_labels):
    """
    Does LCA (Linear Discriminant Analysis) on input matrix with specified number of dimensions. Outputs information
    used to later generate xyz files in the reduced dimensional space and also for the function that filters out
    distances between key atoms and their neighbors.
    :param n_dim: int
    :param matrix: array
    :return: matrix_pca: input data (in matrix) projected onto covariance matrix eigenvectors (PCs)
            matrix_pca_fit: fit used to transform new data into reduced dimensional space
            covar_matrix.components_: eigenvectors of covariance matrix
            covar_matrix.mean_: mean of the original dataset (in matrix)
            covar_matrix.explained_variance_: amount of variance described by each PC
    """

    matrix = pd.DataFrame(matrix)

    lda = discriminant_analysis.LinearDiscriminantAnalysis()

    matrix_lda_fit = lda.fit(matrix, data_labels)
    matrix_lda = lda.transform(matrix)

    return matrix_lda, matrix_lda_fit, lda.components_, lda.mean_, lda.explained_variance_


def calc_mean_distance_vector(d2_matrix):

    return np.mean(d2_matrix, axis=0)


def filter_important_distances(upper_tri_d2_matrices, num_dists=5000):

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
        selected_dist_atom_indexes[i] = atom_indexes[index], index
        i += 1

    print(selected_dist_atom_indexes)

    return important_distances_matrix, selected_dist_atom_indexes


def calc_num_atoms(vec_length):

    n = Symbol('n', positive=True)
    answers = solve(n * (n - 1) / 2 - vec_length, n)
    num_atoms = int(answers[0])

    return num_atoms


def set_unimportant_distance_weights_to_zero(important_distances_matrix, selected_dist_atom_indexes, num_atoms):

    num_points = important_distances_matrix.shape[0]
    num_dists = int((num_atoms*(num_atoms-1))/2)

    all_distances_matrix = np.zeros((num_points, num_dists))

    for i in range(len(selected_dist_atom_indexes)):
        distance_location = selected_dist_atom_indexes[i][1]
        all_distances_matrix[:, distance_location] = important_distances_matrix[:, i]

    return all_distances_matrix


def generate_PC_matrices_selected_distances(n_dim, matrix_reduced, components, mean, selected_dist_atom_indexes, num_atoms):
    #TODO: Figure out what the mean structure should be (i.e., if features have weight of 0 in PC, what's the default?)
    num_points = matrix_reduced.shape[0]
    num_dists = int((num_atoms*(num_atoms-1))/2)

    PCs_separate = []
    for i in range(0, n_dim):
        PCi = np.zeros((num_points, num_dists))
        PCi_selected = np.dot(matrix_reduced[:, i, None], components[None, i, :]) + mean

        for j in range(len(selected_dist_atom_indexes)):
            distance_location = selected_dist_atom_indexes[j][1]
            PCi[:, distance_location] = PCi_selected[:, j]

        PCs_separate.append(PCi)

    PCs_combined = np.zeros((num_points, num_dists))
    PCs_combined_selected = np.dot(matrix_reduced, components) + mean

    for j in range(len(selected_dist_atom_indexes)):
        distance_location = selected_dist_atom_indexes[j][1]
        PCs_combined[:, distance_location] = PCs_combined_selected[:, j]

    PCs_separate = np.array(PCs_separate)
    PCs_combined = np.array(PCs_combined)

    return PCs_separate, PCs_combined


def inverse_transform_of_PCs(n_dim, matrix_reduced, components, mean):

    PCs_separate = []
    for i in range(0, n_dim):
        PCi = np.dot(matrix_reduced[:, i, None], components[None, i, :]) + mean
        PCs_separate.append(PCi)

    PCs_combined = np.dot(matrix_reduced, components) + mean

    PCs_separate = np.array(PCs_separate)
    PCs_combined = np.array(PCs_combined)

    return PCs_separate, PCs_combined

#TODO: Add function for adding linear combinations of eigenvectors to display PCs as normal modes
def generate_eigenvector_matrices_as_normal_modes(n_dim, matrix_reduced, components, mean):

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


def kabsch(coordinates):
    """Kabsch algorithm to get orientation of axes that minimizes RMSD. All structures will be aligned to the first
    structure in the trajectory.
    :param coordinates: coordinates along trajectory to be aligned, list or array
    """
    coordinates = np.array(coordinates)
    coordinates[0] -= rmsd.centroid(coordinates[0])
    coords_kabsch = []
    for i in range(len(coordinates)):
        coordinates[i] -= rmsd.centroid(coordinates[i])
        coords_kabschi = rmsd.kabsch_rotate(coordinates[i], coordinates[0])
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

    return negs, poss, zeros, signs


def chirality_changes(coords_reconstr, stereo_atoms, signs_orig):
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
            # If molecule begins planar but reconstruction of PCs are not, keep chirality consistent along reconstructed
            # trajectory
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
            f.write('%s point %i' % (title, i + 1) + '\n')
            f.write('%s' % str(np.asarray(b)).replace("[", "").replace("]", "").replace("'", "") + '\n')

        f.close()


def plot_prop_of_var(values, name, directory):

    fig = plt.figure(figsize=(8, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)

    normalized_values = values / np.sum(values)
    x = range(len(values))

    ax.scatter(x, normalized_values, c='k')

    ax.set_xlabel("Principal Component", fontsize=16)
    ax.set_ylabel("Proportion of Variance", fontsize=16)
    ax.set_ylim(-0.1, 1.1)

    cumulative = np.cumsum(normalized_values)

    ax1.scatter(x, cumulative)
    ax1.set_xlabel("Principal Component", fontsize=16)
    ax1.set_ylabel("Cumulative Prop. of Var.", fontsize=16)
    ax1.set_ylim(-0.1, 1.1)

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(directory, '%s_proportion_of_variance.png' % name), dpi=600)


def print_prop_of_var_to_txt(values, system_name, directory):

    normalized_values = values / np.sum(values)
    df = pd.DataFrame({'Principal Component': pd.Series([i+1 for i in range(len(values))]),
                       'Singular Value': values,
                       'Prop. of Variance': normalized_values,
                       'Cumul. Prop. of Var.': np.cumsum(normalized_values)})

    pd.set_option('display.expand_frame_repr', False)
    print(df.head())
    df.to_csv(os.path.join(directory, system_name + '_prop_of_var.txt'), sep='\t', index=None)


def print_distance_weights_to_files(directory, n_dim, name, pca_components, num_atoms):

    for n in range(n_dim):
        d = []
        for k in range(len(pca_components[n])):
            i, j = calc_ij(k, num_atoms)
            coeff = pca_components[n][k]
            d.append({'atom 1': i, 'atom 2': j, 'Coefficient of Distance': coeff})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(os.path.join(directory, name + '_PC%s_components.txt' % (n+1)), sep='\t', index=None)


def print_distance_weights_to_files_select_atom_indexes(atom_indexes, n_dim, pca_components, name, directory):

    for n in range(n_dim):
        d = []
        for k in range(len(pca_components[n])):
            coeff = pca_components[n][k]
            d.append({'atom 1': atom_indexes[k][0], 'atom 2': atom_indexes[k][1], 'Coefficient of Distance': coeff})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(os.path.join(directory, name + '_PC%s_components.txt' % (n+1)), sep='\t', index=None)


def print_distance_weights_to_files_weighted(directory, n_dim, name, pca_components, pca_values, num_atoms, display=False):

    for n in range(n_dim):
        d = []
        for k in range(len(pca_components[n])):
            i, j = calc_ij(k, num_atoms)
            coeff = (pca_values[n]/sum(pca_values))*pca_components[n][k]
            d.append({'atom 1': i, 'atom 2': j, 'Coefficient of Distance': coeff})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(os.path.join(directory, name + '_PC%s_components_weighted.txt' % (n+1)), sep='\t', index=None)

        if display:
            print("PC%s" % (n+1))
            print(sorted_d)


def transform_new_data_cartesians(new_input, output_directory, n_dim, pca_fit, pca_components, pca_mean,
                                  original_traj_coords, MW=False):
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
    :param MW: whether coordinates should be mass weighted prior to PCA, bool
    """

    print("\nTransforming %s into reduced dimensional representation..." % new_input)

    new_system_name, atoms, coordinates = read_file(new_input)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("\nResults for %s input will be stored in %s" % (new_input, output_directory))

    # Determining names of output directories/files
    file_name_end = "_Cartesians"

    # Align structures using Kabsch algorithm so rotations don't affect PCs
    aligned_original_traj_coords = kabsch(original_traj_coords)
    coords_for_analysis = align_to_original_traj(coordinates, aligned_original_traj_coords)

    if MW is True:
        file_name_end = file_name_end + "_MW"
        mass_weighted_coords = mass_weighting(atoms, coords_for_analysis)
        coords_for_analysis = mass_weighted_coords

    else:
        file_name_end = file_name_end + "_noMW"
        coords_for_analysis = coords_for_analysis

    coords_for_analysis = np.reshape(coords_for_analysis, (coords_for_analysis.shape[0],
                                                           coords_for_analysis.shape[1] *
                                                           coords_for_analysis.shape[2]))

    components = pca_fit.transform(coords_for_analysis)
    components_df = pd.DataFrame(components)

    PCs_separate = []
    for i in range(0, n_dim):
        PCi = np.dot(components[:, i, None], pca_components[None, i, :]) + pca_mean
        PCs_separate.append(PCi)

    PCs_combined = np.dot(components, pca_components) + pca_mean

    PCs_separate = np.array(PCs_separate)
    PCs_combined = np.array(PCs_combined)

    # Reshape n x 3N x 1 arrays into n x N x 3 arrays
    PCs_separate = np.reshape(PCs_separate, (PCs_separate.shape[0], PCs_separate.shape[1],
                                                 int(PCs_separate.shape[2] / 3), 3))

    PCs_combined = np.reshape(PCs_combined, (1, PCs_combined.shape[0], int(PCs_combined.shape[1] / 3), 3))

    if MW is True:
        # Remove mass-weighting of coordinates
        no_mass_weighting_PCs_separate = [remove_mass_weighting(atoms, PCs_separate[i])
                                          for i in range(n_dim)]
        no_mass_weighting_PCs_combined = remove_mass_weighting(atoms, PCs_combined)
    else:
        no_mass_weighting_PCs_separate = PCs_separate
        no_mass_weighting_PCs_combined = PCs_combined

    aligned_PCs_separate = no_mass_weighting_PCs_separate
    aligned_PCs_combined = no_mass_weighting_PCs_combined

    make_pc_xyz_files(output_directory, new_system_name + file_name_end, atoms, aligned_PCs_separate)
    make_pc_xyz_files(output_directory, new_system_name + file_name_end, atoms, aligned_PCs_combined)

    return new_system_name, components_df


def transform_new_data_distances(new_input, output_directory, n_dim, pca_fit, pca_components, pca_mean,
                                 stereo_atoms=[1, 2, 3, 4], MW=False):
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
    :param MW: whether coordinates should be mass weighted prior to PCA, bool
    """

    print("\nTransforming %s into reduced dimensional representation..." % new_input)

    new_system_name, atoms, coordinates = read_file(new_input)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("\nResults for %s input will be stored in %s" % (new_input, output_directory))

    # Determining names of output directories/files
    file_name_end = "_Distances"

    if MW is True:
        file_name_end = file_name_end + "_MW"
        coordinates_shifted = set_atom_one_to_origin(coordinates)
        mass_weighted_coords = mass_weighting(atoms, coordinates_shifted)
        coords_for_analysis = mass_weighted_coords

    else:
        file_name_end = file_name_end + "_noMW"
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

    # Turning distance matrix representations of structures back into Cartesian coordinates
    PCs_separate = [[distance_matrix_to_coords(PCs_separate[i][k])
                           for k in range(PCs_separate.shape[1])] for i in range(PCs_separate.shape[0])]
    PCs_combined = [distance_matrix_to_coords(PCs_combined[i])
                              for i in range(np.array(PCs_combined).shape[0])]

    PCs_separate = np.real(PCs_separate)
    PCs_combined = np.real(PCs_combined)

    if MW is True:
        # Remove mass-weighting of coordinates
        no_mass_weighting_PCs_separate = [remove_mass_weighting(atoms, PCs_separate[i])
                                          for i in range(n_dim)]
        no_mass_weighting_PCs_combined = remove_mass_weighting(atoms, PCs_combined)
    else:
        no_mass_weighting_PCs_separate = PCs_separate
        no_mass_weighting_PCs_combined = PCs_combined

    # Reorient coordinates so they are in a consistent orientation
    aligned_PCs_separate = [kabsch(chirality_changes(no_mass_weighting_PCs_separate[i], stereo_atoms,
                                                     all_signs)) for i in range(n_dim)]
    aligned_PCs_combined = kabsch(chirality_changes(no_mass_weighting_PCs_combined, stereo_atoms, all_signs))
    aligned_PCs_combined = np.reshape(aligned_PCs_combined, (1, aligned_PCs_combined.shape[0],
                                                  aligned_PCs_combined.shape[1],
                                                  aligned_PCs_combined.shape[2]))

    make_pc_xyz_files(output_directory, new_system_name + file_name_end, atoms, aligned_PCs_separate)
    make_pc_xyz_files(output_directory, new_system_name + file_name_end, atoms, aligned_PCs_combined)

    return new_system_name, components_df


def pathreducer(xyz_file_path, n_dim, stereo_atoms=[1, 2, 3, 4], input_type="Cartesians", MW=False,
                plot_variance=False, print_distance_coefficients=True, reconstruct=True):
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
        name, atoms, coordinates = read_file_df(xyz_file_path)
        num_atoms = len(atoms)
        coords_for_analysis = coordinates

    elif os.path.isdir(xyz_file_path) is True:
        print("\nInput is a directory of files.")
        print("\nDoing dimensionality reduction on files in %s" % xyz_file_path)

        xyz_files = sorted(glob.glob(os.path.join(xyz_file_path, '*.xyz')))

        # Subroutine for if the input specified is a directory of xyz files
        names = []
        atoms = []
        i = 0
        for xyz_file in xyz_files:
            i = i + 1
            name, atoms_one_file, coordinates = read_file_df(xyz_file)
            names.append(name)
            atoms.append(atoms_one_file)
            file_lengths.append(coordinates.shape[0])

            if i == 1:
                coords_for_analysis = coordinates
            else:
                coords_for_analysis = np.concatenate((coords_for_analysis, coordinates), axis=0)

        num_atoms = len(atoms_one_file)

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
            mass_weighted_coordinates = mass_weighting(atoms, coords_for_PCA)
            coords_for_PCA = mass_weighted_coordinates

            print("\n(MW) Done mass-weighting coordinates!")

        coords_for_PCA = np.reshape(coords_for_PCA, (coords_for_PCA.shape[0],
                                                               coords_for_PCA.shape[1] *
                                                               coords_for_PCA.shape[2]))

        # PCA
        cartesians_pca, cartesians_pca_fit, cartesians_components, cartesians_mean, cartesians_values = \
            pca_dr(coords_for_PCA)
        PCs_separate, PCs_combined = inverse_transform_of_PCs(n_dim, cartesians_pca, cartesians_components, cartesians_mean)

        if plot_variance:
            plot_prop_of_var(cartesians_values, name + file_name_end, output_directory)

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
                make_pc_xyz_files(output_directory, name + file_name_end, atoms, no_mass_weighting_PCs_separate)
                make_pc_xyz_files(output_directory, name + file_name_end, atoms, no_mass_weighting_PCs_combined)

                print("\n(4) Done with making output xyz files!")

            elif os.path.isdir(xyz_file_path) is True:
                for x in range(len(file_lengths)):
                    filename = names[x]
                    if x == 0:
                        start_index = 0
                        end_index = file_lengths[x]
                        one_file_PCs_separate = np.array(no_mass_weighting_PCs_separate)[:, start_index:end_index, :, :]
                        one_file_PCs_combined = np.array(no_mass_weighting_PCs_combined)[:, start_index:end_index, :, :]
                        make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                        make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_combined)
                    else:
                        start_index = sum(file_lengths[:x])
                        end_index = sum(file_lengths[:(x + 1)])
                        one_file_PCs_separate = np.array(no_mass_weighting_PCs_separate)[:, start_index:end_index, :, :]
                        one_file_PCs_combined = np.array(no_mass_weighting_PCs_combined)[:, start_index:end_index, :, :]
                        make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                        make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_combined)

            print("\nDone generating output!")

        return name, output_directory, cartesians_pca, cartesians_pca_fit, cartesians_components, \
               cartesians_mean, cartesians_values, file_lengths, coords_for_analysis

    elif input_type == "Distances":

        aligned_coordinates = kabsch(coordinates)

        if MW is True:
            coordinates_shifted = set_atom_one_to_origin(coordinates)
            mass_weighted_coordinates = mass_weighting(atoms, coordinates_shifted)
            coords_for_PCA = mass_weighted_coordinates

            print("\n(MW) Done mass-weighting coordinates!")

        else:
            coords_for_PCA = coords_for_analysis

        negatives, positives, zeroes, all_signs = chirality_test(coords_for_analysis, stereo_atoms)

        if coords_for_PCA.shape[1] > 1000:
            num_dists = 75000
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
        d_pca, d_pca_fit, d_components, d_mean, d_values = pca_dr(d_re)

        print("\n(2) Done with PCA of %s!" % input_type)

        if coords_for_PCA.shape[1] > 1000:
            # d_re = set_unimportant_distance_weights_to_zero(d_re, selected_dist_atom_indexes, num_atoms)
            PCs_separate_d, PCs_combined_d = generate_PC_matrices_selected_distances(n_dim, d_pca, d_components, d_mean,
                                                                                     selected_dist_atom_indexes, num_atoms)
        else:
            PCs_separate_d, PCs_combined_d = inverse_transform_of_PCs(n_dim, d_pca, d_components, d_mean)

        print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")

        if plot_variance:
            plot_prop_of_var(d_values, name + file_name_end, output_directory)

        if print_distance_coefficients:
            if coords_for_PCA.shape[1] > 1000:
                print_distance_weights_to_files_select_atom_indexes(selected_dist_atom_indexes, n_dim, d_components,
                                                                    name + file_name_end, output_directory)
            else:
                print_distance_weights_to_files(output_directory, n_dim, name + file_name_end, d_components, len(atoms))

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
                no_mass_weighting_PCs_separate = [remove_mass_weighting(atoms, PCs_separate[i])
                                                  for i in range(n_dim)]
                no_mass_weighting_PCs_combined = remove_mass_weighting(atoms, PCs_combined)
                print("\n(UMW) Done removing mass-weighting!")

            else:
                no_mass_weighting_PCs_separate = PCs_separate
                no_mass_weighting_PCs_combined = PCs_combined

            chirality_consistent_PCs_separate = [chirality_changes(no_mass_weighting_PCs_separate[i], stereo_atoms, all_signs)
                                                 for i in range(n_dim)]

            # Reorient coordinates so they are in a consistent coordinate system/chirality, all Xs combined into one array
            chirality_consistent_PCs_combined = kabsch(chirality_changes(no_mass_weighting_PCs_combined, stereo_atoms, all_signs))

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
                make_pc_xyz_files(output_directory, name + file_name_end, atoms, aligned_PCs_separate)
                make_pc_xyz_files(output_directory, name + file_name_end, atoms, aligned_PCs_combined)

            elif os.path.isdir(xyz_file_path) is True:
                for x in range(len(file_lengths)):
                    filename = names[x]
                    if x == 0:
                        start_index = 0
                        end_index = file_lengths[x]
                        one_file_PCs_separate = np.array(aligned_PCs_separate)[:, start_index:end_index, :, :]
                        one_file_PCs_combined = np.array(aligned_PCs_combined)[:, start_index:end_index, :, :]
                        make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                        make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_combined)
                    else:
                        start_index = sum(file_lengths[:x])
                        end_index = sum(file_lengths[:(x + 1)])
                        one_file_PCs_separate = np.array(aligned_PCs_separate)[:, start_index:end_index, :, :]
                        one_file_PCs_combined = np.array(aligned_PCs_combined)[:, start_index:end_index, :, :]
                        make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                        make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_combined)

        print("\nDone generating output!")

        return name, output_directory, d_pca, d_pca_fit, d_components, d_mean, d_values, file_lengths, aligned_coordinates


def pathreducer_cartesians_one_file(xyz_file_path, n_dim, MW=False, reconstruct=True):
    """
    Workhorse function for doing dimensionality reduction on xyz files. Dimensionality reduction can be done on the
    structures represented as Cartesian coordinates (easy/faster) or the structures represented as distances matrices
    (slower, but potentially more useful for certain systems that vary in non-linear ways, e.g., torsions).
    :param xyz_file_path: xyz file or directory filled with xyz files that will be used to generate the reduced dimensional
    space, str
    :param n_dim: number of dimensions to reduce system to using PCA, int
    :return: name, directory, pca, pca_fit, components, mean, values, lengths
    """

    # Make sure even large matrices are printed out in their entirety (for the generation of xyz files)
    np.set_printoptions(threshold=np.nan)

    # Check if input is directory (containing input files) or a single input file itself
    assert os.path.isfile(xyz_file_path) or os.path.isdir(xyz_file_path), "No such file or directory."

    # Determining names of output directories/files
    file_name_end = "_Cartesians"

    if MW is True:
        file_name_end = file_name_end + "_MW"
    elif MW is False:
        file_name_end = file_name_end + "_noMW"

    print("\nInput is one file.")
    system_name, atoms, coordinates = read_file(xyz_file_path)

    # Creating a directory for output (if directory doesn't already exist)
    output_directory = system_name + file_name_end + "_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Results for %s input will be stored in %s" % (xyz_file_path, output_directory))

    aligned_coords = kabsch(coordinates)
    print("\n(1C) Done aligning structures using Kabsch algorithm")

    if MW is True:
        mass_weighted_coordinates = mass_weighting(atoms, aligned_coords)
        print("\n(MW) Done mass-weighting coordinates!")

        matrix_for_pca = np.reshape(mass_weighted_coordinates, (mass_weighted_coordinates.shape[0],
                                                               mass_weighted_coordinates.shape[1] *
                                                               mass_weighted_coordinates.shape[2]))
    else:
        matrix_for_pca = np.reshape(aligned_coords, (aligned_coords.shape[0], aligned_coords.shape[1] *
                                                     aligned_coords.shape[2]))

    # PCA
    cartesians_pca, cartesians_pca_fit, cartesians_components, cartesians_mean, cartesians_values = \
        pca_dr(matrix_for_pca)

    print("\n(2) Done with PCA of Cartesian coordinates!")

    if reconstruct:
        PCs_separate, PCs_combined = inverse_transform_of_PCs(n_dim, cartesians_pca, cartesians_components,
                                                              cartesians_mean)

        print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")

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
        make_pc_xyz_files(output_directory, system_name + file_name_end, atoms, no_mass_weighting_PCs_separate)
        make_pc_xyz_files(output_directory, system_name + file_name_end, atoms, no_mass_weighting_PCs_combined)

        print("\n(4) Done with making output xyz files!")

    return system_name, output_directory, cartesians_pca, cartesians_pca_fit, cartesians_components, cartesians_mean, \
           cartesians_values, aligned_coords


def pathreducer_cartesians_directory_of_files(xyz_file_directory_path, n_dim, MW=False, reconstruct=True):
    """
    Workhorse function for doing dimensionality reduction on xyz files. Dimensionality reduction can be done on the
    structures represented as Cartesian coordinates (easy/faster) or the structures represented as distances matrices
    (slower, but potentially more useful for certain systems that vary in non-linear ways, e.g., torsions).
    :param xyz_file_directory_path: xyz file or directory filled with xyz files that will be used to generate the
    reduced dimensional space, str
    :param n_dim: number of dimensions to reduce system to using PCA, int
    :return: name, directory, pca, pca_fit, components, mean, values, lengths
    """

    # Make sure even large matrices are printed out in their entirety (for the generation of xyz files)
    np.set_printoptions(threshold=np.nan)

    # Check if input is directory (containing input files) or a single input file itself
    assert os.path.isfile(xyz_file_directory_path) or os.path.isdir(xyz_file_directory_path), "No such file or directory."

    # Determining names of output directories/files
    file_name_end = "_Cartesians"

    if MW is True:
        file_name_end = file_name_end + "_MW"
    elif MW is False:
        file_name_end = file_name_end + "_noMW"

    print("\nInput is a directory of files.")

    path = os.path.dirname(xyz_file_directory_path)
    system_name = os.path.basename(path)
    print("\nDoing dimensionality reduction on files in %s" % system_name)

    xyz_files = sorted(glob.glob(os.path.join(xyz_file_directory_path, '*.xyz')))

    names = []
    atoms = []
    file_lengths = []
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

    # Creating a directory for output (if directory doesn't already exist)
    output_directory = system_name + file_name_end + "_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Results for %s input will be stored in %s" % (xyz_file_directory_path, output_directory))

    aligned_coords = kabsch(coords_for_analysis)
    print("\n(1C) Done aligning structures using Kabsch algorithm")

    if MW is True:
        mass_weighted_coordinates = mass_weighting(atoms_one_file, aligned_coords)
        print("\n(MW) Done mass-weighting coordinates!")

        matrix_for_pca = np.reshape(mass_weighted_coordinates, (mass_weighted_coordinates.shape[0],
                                    mass_weighted_coordinates.shape[1] * mass_weighted_coordinates.shape[2]))

    else:
        matrix_for_pca = np.reshape(aligned_coords, (aligned_coords.shape[0],
                                                           aligned_coords.shape[1] * aligned_coords.shape[2]))

    # PCA
    cartesians_pca, cartesians_pca_fit, cartesians_components, cartesians_mean, cartesians_values = \
        pca_dr(matrix_for_pca)
    print("\n(2) Done with PCA of Cartesian coordinates!")

    if reconstruct:
        PCs_separate, PCs_combined = inverse_transform_of_PCs(n_dim, cartesians_pca, cartesians_components,
                                                              cartesians_mean)
        print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")

        # Reshape n x 3N x 1 arrays into n x N x 3 arrays
        PCs_separate = np.reshape(PCs_separate, (PCs_separate.shape[0], PCs_separate.shape[1],
                                                     int(PCs_separate.shape[2] / 3), 3))

        PCs_combined = np.reshape(PCs_combined, (1, PCs_combined.shape[0], int(PCs_combined.shape[1] / 3), 3))

        if MW is True:
            # Remove mass-weighting of coordinates, individual Xs
            no_mass_weighting_PCs_separate = [remove_mass_weighting(atoms_one_file, PCs_separate[i]) for i in range(n_dim)]

            # Remove mass-weighting of coordinates, all Xs combined into one array/reduced dimensional trajectory
            no_mass_weighting_PCs_combined = remove_mass_weighting(atoms_one_file, PCs_combined)
            print("\n(UMW) Done removing mass-weighting!")

        else:
            no_mass_weighting_PCs_separate = [PCs_separate[i] for i in range(n_dim)]
            no_mass_weighting_PCs_combined = PCs_combined

        # Make xyz files from final coordinate arrays
        for x in range(len(file_lengths)):
            filename = names[x]
            if x == 0:
                start_index = 0
                end_index = file_lengths[x]
                one_file_PCs_separate = np.array(no_mass_weighting_PCs_separate)[:, start_index:end_index, :, :]
                one_file_PCs_combined = np.array(no_mass_weighting_PCs_combined)[:, start_index:end_index, :, :]
                make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_combined)
            else:
                start_index = sum(file_lengths[:x])
                end_index = sum(file_lengths[:(x + 1)])
                one_file_PCs_separate = np.array(no_mass_weighting_PCs_separate)[:, start_index:end_index, :, :]
                one_file_PCs_combined = np.array(no_mass_weighting_PCs_combined)[:, start_index:end_index, :, :]
                make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_combined)

        print("\nDone generating output!")

    return system_name, output_directory, cartesians_pca, cartesians_pca_fit, cartesians_components, cartesians_mean, \
           cartesians_values, file_lengths, coords_for_analysis


def pathreducer_distances_one_file(xyz_file_path, n_dim, stereo_atoms=[1, 2, 3, 4], MW=False,
                                   print_distance_coefficients=True, reconstruct=True):
    """
    Workhorse function for doing dimensionality reduction on xyz files. Dimensionality reduction can be done on the
    structures represented as Cartesian coordinates (easy/faster) or the structures represented as distances matrices
    (slower, but potentially more useful for certain systems that vary in non-linear ways, e.g., torsions).
    :param xyz_file_path: xyz file or directory filled with xyz files that will be used to generate the reduced dimensional
    space, str
    :param n_dim: number of dimensions to reduce system to using PCA, int
    :param stereo_atoms: list of 4 atom indexes surrounding stereogenic center, ints
    :return: name, directory, pca, pca_fit, components, mean, values, lengths
    """

    # Make sure even large matrices are printed out in their entirety (for the generation of xyz files)
    np.set_printoptions(threshold=np.nan)

    # Check if input is directory (containing input files) or a single input file itself
    assert os.path.isfile(xyz_file_path) or os.path.isdir(xyz_file_path), "No such file or directory."

    # Determining names of output directories/files
    file_name_end = "_Distances"
    if MW is True:
        file_name_end = file_name_end + "_MW"
    elif MW is False:
        file_name_end = file_name_end + "_noMW"

    print("\nInput is one file.")
    name, atoms, coordinates = read_file(xyz_file_path)

    # Creating a directory for output (if directory doesn't already exist)
    output_directory = name + file_name_end + "_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Results for %s input will be stored in %s" % (xyz_file_path, output_directory))

    aligned_coordinates = kabsch(coordinates)
    negatives, positives, zeroes, all_signs = chirality_test(aligned_coordinates, stereo_atoms)

    if MW is True:
        coordinates_shifted = set_atom_one_to_origin(coordinates)
        mass_weighted_coordinates = mass_weighting(atoms, coordinates_shifted)
        coords_for_pca = mass_weighted_coordinates

        print("\n(MW) Done mass-weighting coordinates!")

    else:
        coords_for_pca = aligned_coordinates

    d2_full_matrices = generate_distance_matrices(coords_for_pca)
    d2_vector_matrix = reshape_ds(d2_full_matrices)
    print("\n(1D) Generation of distance matrices and reshaping upper triangles into vectors done!")

    # PCA on distance matrix
    d_pca, d_pca_fit, d_components, d_mean, d_values = pca_dr(d2_vector_matrix)
    print("\n(2) Done with PCA of structures as distance matrices!")

    PCs_separate_d, PCs_combined_d = inverse_transform_of_PCs(n_dim, d_pca, d_components, d_mean)
    print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")

    if print_distance_coefficients:
        print_distance_weights_to_files(output_directory, n_dim, name + file_name_end, d_components, len(atoms))

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
            no_mass_weighting_PCs_separate = [remove_mass_weighting(atoms, PCs_separate[i])
                                              for i in range(n_dim)]
            no_mass_weighting_PCs_combined = remove_mass_weighting(atoms, PCs_combined)
            print("\n(UMW) Done removing mass-weighting!")

        else:
            no_mass_weighting_PCs_separate = PCs_separate
            no_mass_weighting_PCs_combined = PCs_combined

        chirality_consistent_PCs_separate = [chirality_changes(no_mass_weighting_PCs_separate[i], stereo_atoms,
                                                               all_signs) for i in range(n_dim)]

        # Reorient coordinates so they are in a consistent coordinate system/chirality, all Xs combined into one array
        chirality_consistent_PCs_combined = kabsch(chirality_changes(no_mass_weighting_PCs_combined, stereo_atoms,
                                                                     all_signs))

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

        # Make final structures into xyz files
        make_pc_xyz_files(output_directory, name + file_name_end, atoms, aligned_PCs_separate)
        make_pc_xyz_files(output_directory, name + file_name_end, atoms, aligned_PCs_combined)

    print("\nDone generating output!")

    return name, output_directory, d_pca, d_pca_fit, d_components, d_mean, d_values, aligned_coordinates


def pathreducer_distances_directory_of_files(xyz_file_directory_path, n_dim, stereo_atoms=[1, 2, 3, 4], MW=False,
                                             print_distance_coefficients=True, reconstruct=True):
    """
    Workhorse function for doing dimensionality reduction on xyz files. Dimensionality reduction can be done on the
    structures represented as Cartesian coordinates (easy/faster) or the structures represented as distances matrices
    (slower, but potentially more useful for certain systems that vary in non-linear ways, e.g., torsions).
    :param xyz_file_directory_path: xyz file or directory filled with xyz files that will be used to generate the reduced dimensional
    space, str
    :param n_dim: number of dimensions to reduce system to using PCA, int
    :param stereo_atoms: list of 4 atom indexes surrounding stereogenic center, ints
    :return: name, directory, pca, pca_fit, components, mean, values, lengths
    """
    # Check if input is directory (containing input files) or a single input file itself
    assert os.path.isfile(xyz_file_directory_path) or os.path.isdir(xyz_file_directory_path), "No such file or directory."

    print("\nInput is a directory of files.")

    # Make sure even large matrices are printed out in their entirety (for the generation of xyz files)
    np.set_printoptions(threshold=np.nan)

    # Determining names of output directories/files
    file_name_end = "_Distances"
    if MW is True:
        file_name_end = file_name_end + "_MW"
    elif MW is False:
        file_name_end = file_name_end + "_noMW"

    path = os.path.dirname(xyz_file_directory_path)
    system_name = os.path.basename(path)
    print("\nDoing dimensionality reduction on files in %s" % system_name)

    xyz_files = sorted(glob.glob(os.path.join(xyz_file_directory_path, '*.xyz')))

    names = []
    atoms = []
    file_lengths = []
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

    # Creating a directory for output (if directory doesn't already exist)
    output_directory = system_name + file_name_end + "_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Results for %s input will be stored in %s" % (system_name, output_directory))

    aligned_coordinates = kabsch(coords_for_analysis)
    negatives, positives, zeroes, all_signs = chirality_test(coords_for_analysis, stereo_atoms)

    if MW is True:
        coordinates_shifted = set_atom_one_to_origin(coords_for_analysis)
        mass_weighted_coordinates = mass_weighting(atoms_one_file, coordinates_shifted)
        print("\n(MW) Done mass-weighting coordinates!")
        coords_for_pca = mass_weighted_coordinates

    else:
        coords_for_pca = coords_for_analysis

    d2_full_matrices = generate_distance_matrices(coords_for_pca)
    d2_vector_matrix = reshape_ds(d2_full_matrices)

    print("\n(1D) Generation of distance matrices and reshaping upper triangles into vectors done!")

    # PCA on distance matrix
    d_pca, d_pca_fit, d_components, d_mean, d_values = pca_dr(d2_vector_matrix)

    print("\n(2) Done with PCA of structures as interatomic distance matrices!")

    PCs_separate_d, PCs_combined_d = inverse_transform_of_PCs(n_dim, d_pca, d_components, d_mean)

    print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")

    if print_distance_coefficients:
        print_distance_weights_to_files(output_directory, n_dim, system_name + file_name_end, d_components,
                                        len(atoms_one_file))

    if reconstruct:
        # Turning distance matrix representations of structures back into Cartesian coordinates
        PCs_separate = [[distance_matrix_to_coords(PCs_separate_d[i][k])
                               for k in range(PCs_separate_d.shape[1])] for i in range(PCs_separate_d.shape[0])]
        # Turning distance matrix representations of structures back into Cartesian coordinates (all chosen PCs combined
        # into one xyz file)
        PCs_combined = [distance_matrix_to_coords(PCs_combined_d[i]) for i in range(np.array(PCs_combined_d).shape[0])]

        PCs_separate = np.real(PCs_separate)
        PCs_combined = np.real(PCs_combined)

        print("\n(4D)-(6D) Done with converting distance matrices back to Cartesian coordinates!")

        if MW is True:
            # Remove mass-weighting of coordinates, individual PCs
            no_mass_weighting_PCs_separate = [remove_mass_weighting(atoms_one_file, PCs_separate[i])
                                              for i in range(n_dim)]
            no_mass_weighting_PCs_combined = remove_mass_weighting(atoms_one_file, PCs_combined)
            print("\n(UMW) Done removing mass-weighting!")

        else:
            no_mass_weighting_PCs_separate = PCs_separate
            no_mass_weighting_PCs_combined = PCs_combined

        chirality_consistent_PCs_separate = [chirality_changes(no_mass_weighting_PCs_separate[i], stereo_atoms, all_signs)
                                             for i in range(n_dim)]

        # Reorient coordinates so they are in a consistent coordinate system/chirality, all Xs combined into one array
        chirality_consistent_PCs_combined = kabsch(chirality_changes(no_mass_weighting_PCs_combined, stereo_atoms, all_signs))

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

        for x in range(len(xyz_files)):
            filename = names[x]
            if x == 0:
                start_index = 0
                end_index = file_lengths[x]
                one_file_PCs_separate = np.array(aligned_PCs_separate)[:, start_index:end_index, :, :]
                one_file_PCs_combined = np.array(aligned_PCs_combined)[:, start_index:end_index, :, :]
                make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_combined)
            else:
                start_index = sum(file_lengths[:x])
                end_index = sum(file_lengths[:(x + 1)])
                one_file_PCs_separate = np.array(aligned_PCs_separate)[:, start_index:end_index, :, :]
                one_file_PCs_combined = np.array(aligned_PCs_combined)[:, start_index:end_index, :, :]
                make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_separate)
                make_pc_xyz_files(output_directory, filename + file_name_end, atoms_one_file, one_file_PCs_combined)

    print("\nDone generating output!")

    return system_name, output_directory, d_pca, d_pca_fit, d_components, d_mean, d_values, file_lengths, \
           aligned_coordinates


def pathreducer_interactive():

    while True:
        input_path = input("\nInput a path to an xyz file or directory of xyz files.\n")

        if os.path.isfile(input_path):
            print("Input is an individual file.")
            system_name = os.path.basename(input_path)
            break

        elif os.path.isdir(input_path):
            print("Input is a directory of files.")
            path = os.path.dirname(input_path)
            system_name = os.path.basename(path)
            break

        else:
            print("No such file or directory.")
            continue

    print("\nDoing dimensionality reduction on files in %s" % system_name)

    while True:
        mass_weight = input("\nWould you like to mass-weight the Cartesian coordinates of your structures prior to "
                            "dimensionality reduction? (True or False)\n")
        if mass_weight in ("True", "true", "T", "t"):
            mw = True
            break
        elif mass_weight in ("False", "false", "F", "f"):
            mw = False
            break
        else:
            print("Please type True or False.")
            continue

    while True:
        structure_type = input("\nHow would you like to represent your structures? (Cartesians or Distances)\n")
        if structure_type in ("Cartesians", "cartesians", "C", "c"):
            input_type = "Cartesians"
            break
        elif structure_type in ("Distances", "distances", "D", "d"):
            input_type = "Distances"

            while True:
                stereo_atoms = input(
                    "\nOptional: Enter four atom numbers (separated by commas) to define the chirality of your "
                    "molecule. Hit Return to skip.\n")

                if stereo_atoms == "":
                    stereo_atoms = '1, 2, 3, 4'
                    break

                elif len(stereo_atoms.split(',')) == 4:
                    break

                elif len(stereo_atoms.split(',')) != 4:
                    print("Please enter four atom numbers separated by commas, or hit Return to skip.")
                    continue

            stereo_atoms = [int(s) for s in stereo_atoms.split(',')]

            break
        else:
            print("Please type Cartesians or Distances.")
            continue

    while True:
        try:
            n_dim = int(input("\nHow many principal components would you like to print out? "
                              "(If you're not sure, use 3)\n"))
        except ValueError:
            print("Sorry, number of principal components must be an integer value.")
            continue

        if n_dim <= 0:
            print("Sorry, number of principal components must be greater than zero.")
            continue
        else:
            break

    if os.path.isfile(input_path) and input_type == "Cartesians":
        system_name, output_directory, pca, pca_fit, components, mean, singular_values, aligned_coords = \
            pathreducer_cartesians_one_file(input_path, n_dim, MW=mw)
    elif os.path.isdir(input_path) and input_type == "Cartesians":
        system_name, output_directory, pca, pca_fit, components, mean, singular_values, traj_lengths, aligned_coords = \
            pathreducer_cartesians_directory_of_files(input_path, n_dim, MW=mw)
    elif os.path.isfile(input_path) and input_type == "Distances":
        system_name, output_directory, pca, pca_fit, components, mean, singular_values, aligned_coords = \
            pathreducer_distances_one_file(input_path, n_dim,stereo_atoms=stereo_atoms, MW=mw)
    elif os.path.isdir(input_path) and input_type == "Distances":
        system_name, output_directory, pca, pca_fit, components, mean, singular_values, traj_lengths, aligned_coords = \
            pathreducer_distances_directory_of_files(input_path, n_dim, stereo_atoms=stereo_atoms, MW=mw)
    else:
        print("Something went wrong.")

    pcs_df = pd.DataFrame(pca)
    if os.path.isdir(input_path):
        lengths = traj_lengths
    else:
        lengths = None

    plot_variance = input("\nWould you like a plot of the variance captured by each PC? (True or False)\n")
    if plot_variance == "True":
        plot_prop_of_var(singular_values, system_name, output_directory)
        print_prop_of_var_to_txt(singular_values, system_name, output_directory)

    plot_pcs = input("\nWould you like a plot of the top two and top three PCs? (True or False)\n")
    if plot_pcs == "True":
        points_to_circle = input("\nIf you have points to circle, enter them now, separated by commas.\n")
        if points_to_circle != "":
            points_to_circle = [int(s) for s in points_to_circle.split(',')]
        else:
            points_to_circle = None
        plotting_functions.colored_line_and_scatter_plot(pcs_df[0], pcs_df[1], pcs_df[2], same_axis=False,
                                                         output_directory=output_directory, lengths=lengths,
                                                         points_to_circle=points_to_circle,
                                                         imgname=(system_name + "_scatterline"))

    new_data_to_project = input("\nDo you have new data you would like to project into this reduced dimensional space? "
                     "(True or False)\n")

    while new_data_to_project == "True":
        new_input = input("\nWhat is the path to the file of interest? (Can only take one file at a time)\n")
        if input_type == "Cartesians":
            new_system_name, new_data_df = transform_new_data_cartesians(new_input, output_directory, n_dim, pca_fit, components, mean,
                                      aligned_coords, MW=mw)
        elif input_type == "Distances":
            new_system_name, new_data_df = transform_new_data_distances(new_input, output_directory, n_dim, pca_fit, components, mean,
                                         stereo_atoms=stereo_atoms, MW=mw)
        plot_new_data = input("\nWould you like to plot the new data? (True or False)\n")
        if plot_new_data == "True":
            plotting_functions.colored_line_and_scatter_plot(pcs_df[0], pcs_df[1], pcs_df[2], same_axis=False,
                                                             output_directory=output_directory, lengths=lengths,
                                                             new_data=new_data_df, points_to_circle=points_to_circle,
                                                             imgname=(new_system_name + "_scatterline"))
        continue


def generate_deformation_vector(start_structure, end_structure):

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    nmd_coords = np.reshape(start_structure, (1, np.array(start_structure).shape[0]*np.array(start_structure).shape[1]))

    deformation_vector = end_structure - start_structure
    deformation_vector = np.reshape(deformation_vector,
                                    (1, np.array(deformation_vector).shape[0]*np.array(deformation_vector).shape[1]))
    print("NMD Coordinates:", nmd_coords)
    print("Deformation vector:", deformation_vector)

    return deformation_vector
