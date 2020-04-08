#!/usr/bin/env python
# coding: utf-8

import numpy as np
import calculate_rmsd as rmsd
import pandas as pd
import math
import glob
import os
import sys
import ntpath
import plotting_functions
import MDAnalysis
from periodictable import *
from sklearn import *
from sympy import solve, Symbol


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def read_xyz_file(path):
    """ Reads in an xyz file from path as a DataFrame. This DataFrame is then turned into a 3D array such that the
    dimensions are (number of points) X (number of atoms) X 3 (Cartesian coordinates). The system name (based on the
    filename), list of atoms in the system, and Cartesian coordinates are output.
    :param path: path to xyz file to be read
    :return atom_list: numpy array
            cartesians: numpy array
    """
    data = pd.read_csv(path, header=None, delim_whitespace=True, names=['atom', 'X', 'Y', 'Z'])
    n_atoms = int(data.loc[0][0])
    n_lines_per_frame = int(n_atoms + 2)

    data_array = np.array(data)

    data_reshape = np.reshape(data_array, (int(data_array.shape[0]/n_lines_per_frame), n_lines_per_frame,
                                           data_array.shape[1]))
    cartesians = data_reshape[:, 2::, 1::].astype(np.float)
    atom_list = data_reshape[0, 2::, 0]

    return atom_list, cartesians


def read_gamess_file(path):
    """ Reads in a GAMESS trj file from path using MDAnalysis. This DataFrame is then turned into a 3D array such that the
    dimensions are (number of points) X (number of atoms) X 3 (Cartesian coordinates). The system name (based on the
    filename), list of atoms in the system, and Cartesian coordinates are output.
    :param path: path to xyz file to be read
    :return extensionless_system_name: str
            atom_list: numpy array
            cartesians: numpy array
    """

    data = pd.read_csv(path, header=None, delim_whitespace=True, names=['a', 'b', 'c', 'd', 'e', 'f'])
    n_atoms = int(data.loc[1][1])
    n_lines_per_frame = int(n_atoms * 2 + 6)

    data_array = np.array(data)

    data_reshape = np.reshape(data_array, (int(data_array.shape[0]/n_lines_per_frame), n_lines_per_frame,
                                           data_array.shape[1]))

    cartesians = data_reshape[:, 5:(n_atoms + 5), 2:5].astype(np.float)
    atom_list = data_reshape[0, 5:(n_atoms + 5), 0]

    return atom_list, cartesians


def determine_type_and_read_file(path):

    system_name = path_leaf(path)
    print("File being read is: %s" % system_name)

    extensionless_system_name = os.path.splitext(system_name)[0]

    if system_name.endswith('.xyz'):
        atom_list, cartesians = read_xyz_file(path)

    elif system_name.endswith('.trj'):
        atom_list, cartesians = read_gamess_file(path)

    return extensionless_system_name, atom_list, cartesians


def remove_atoms_by_type(atom_types_to_remove, atom_list, cartesians):
    """
    Removes specific atoms if they are not wanted for PCA
    :param atom_list: list of atoms in the structure
    :param cartesians: cartesian coordinates of each frame
    :return: cartesian coordinates of each frame with specific atom types removed
    """
    matches_indexes = [i for i, x in enumerate(atom_list) if x in atom_types_to_remove]
    cartesians_sans_atoms = np.delete(cartesians, list(matches_indexes), axis=1)
    atom_list_sans_atoms = np.delete(atom_list, list(matches_indexes), axis=0)

    return atom_list_sans_atoms, cartesians_sans_atoms


def include_solvent_shell(atom_list, solute_indexes, cartesians, radius=10.0):

    num_atoms = len(atom_list)
    atom_indexes_within_radius = []
    distances = []
    for index_a in solute_indexes:
        for index_b in range(num_atoms):
            if index_b not in solute_indexes:
                # print(index_b)
                # Calculate only distances between atoms with the most variance and all other atoms. Only keep
                # distances less than x angstroms (default = 7.0) at the beginning (frame 0) of the trajectory
                a = cartesians[0][index_a]
                b = cartesians[0][index_b]
                d = np.linalg.norm(a - b)
                if d < radius and index_b not in atom_indexes_within_radius:
                    # print("A: %s, B: %s, dist: %s" % (index_a, index_b, d))
                    distances.append(d)
                    atom_indexes_within_radius.append(index_b)

    # Generate sorted list of indexes of atoms involved the most in PC1 and all atoms within the distance threshold
    top_atom_indexes = list(solute_indexes) + list(atom_indexes_within_radius)
    top_atom_indexes.sort()

    cartesians_solute_plus_solvent_shell = cartesians[:, top_atom_indexes, :]

    return cartesians_solute_plus_solvent_shell


def calculate_velocities(cartesians, timestep=1):
    """
    Calculate velocities at each timestep given Cartesian coordinates. Velocities at the first and last point are
    extrapolated.
    :param cartesians: Cartesian coordinates along trajectory
    :param timestep: time step between frames in units of fs, default=1
    :return: velocities
    """

    velocities = []
    for i in range(0, len(cartesians)):
        if i == 0:
            velocity = (cartesians[i + 1] - cartesians[i]) / timestep
        elif i == len(cartesians)-1:
            velocity = (cartesians[i] - cartesians[i - 1]) / timestep
        else:
            velocity = (cartesians[i + 1] - cartesians[i - 1])/2*timestep

        velocities.append(velocity)

    return velocities


def calculate_momenta(velocities, atoms):
    """

    :param cartesians: Cartesian coordinates along trajectory
    :param timestep: time step between frames in units of fs, default=1
    :return: velocities
    """
    velocities = np.array(velocities)
    atoms = np.array(atoms)

    atom_masses = np.array([formula(atom).mass for atom in atoms])
    momenta = velocities * atom_masses[np.newaxis, :, np.newaxis]

    return momenta


def set_atom_one_to_origin(coordinates):
    coordinates_shifted = coordinates - coordinates[:, np.newaxis, 0]

    return coordinates_shifted


def mass_weighting(atoms, cartesians):

    cartesians = np.array(cartesians)
    atoms = np.array(atoms)

    atom_masses = [formula(atom).mass for atom in atoms]
    weighting = np.sqrt(atom_masses)
    mass_weighted_cartesians = cartesians * weighting[np.newaxis, :, np.newaxis]

    return mass_weighted_cartesians


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


def calculate_dihedral(indexes, cartesians):
    """
    Calculates the dihedral angle between four atom indexes
    :param indexes: list of atom indexes between which to calculate the angle. Four ints long.
    :param cartesians: n x N x 3 array of Cartesian coordinates along the course of a trajectory.
    :return: dihedral: n x 1 array of specified dihedral angle along course of trajectory
    """
    B1 = cartesians[:, indexes[1]] - cartesians[:, indexes[0]]
    B2 = cartesians[:, indexes[2]] - cartesians[:, indexes[1]]
    B3 = cartesians[:, indexes[3]] - cartesians[:, indexes[2]]

    modB2 = np.sqrt((B2[:, 0] ** 2) + (B2[:, 1] ** 2) + (B2[:, 2] ** 2))

    yAx = modB2 * B1[:, 0]
    yAy = modB2 * B1[:, 1]
    yAz = modB2 * B1[:, 2]

    # CP2 is the crossproduct of B2 and B3
    CP2 = np.cross(B2, B3)

    termY = (yAx * CP2[:, 0]) + (yAy * CP2[:, 1]) + (yAz * CP2[:, 2])

    # CP is the crossproduct of B1 and B2
    CP = np.cross(B1, B2)

    termX = (CP[:, 0] * CP2[:, 0]) + (CP[:, 1] * CP2[:, 1]) + (CP[:, 2] * CP2[:, 2])

    dihedral = (180 / np.pi) * np.arctan2(termY, termX)

    return dihedral


def generate_dihedral_matrices(coordinates):

    

    return coordinates


def generate_and_reshape_ds_big_structures(coordinates):
    """ Generates matrix of pairwise distances, which includes pairwise distances for each structure.
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


def pca_dr(matrix, return_covariance=False):
    """
    Does PCA on input matrix with specified number of dimensions. Outputs information used to later generate xyz files
    in the reduced dimensional space and also for the function that filters out distances between key atoms and their
    neighbors.
    :param matrix: array
    :return: matrix_pca: input data (in matrix) projected onto covariance matrix eigenvectors (PCs)
            matrix_pca_fit: fit used to transform new data into reduced dimensional space
            pca.components_: eigenvectors of covariance matrix
            pca.mean_: mean of the original dataset (in matrix)
            pca.explained_variance_: amount of variance described by each PC
    """

    matrix = pd.DataFrame(matrix)

    pca = decomposition.PCA()

    matrix_pca_fit = pca.fit(matrix)
    matrix_pca = pca.transform(matrix)

    if return_covariance:
        covariance_matrix = pca.get_covariance()


    return matrix_pca, matrix_pca_fit, pca.components_, pca.mean_, pca.explained_variance_, \
           covariance_matrix if 'covariance_matrix' in locals() else None


#TODO: Add function that is able to do LDA on data rather than PCA
def lda_dr(matrix, data_labels):
    """
    Does LDA (Linear Discriminant Analysis) on input matrix with specified number of dimensions. Outputs information
    used to later generate xyz files in the reduced dimensional space and also for the function that filters out
    distances between key atoms and their neighbors.
    :param matrix: array
    :param data_labels: array, separates data into classes to differentiate
    :return: matrix_lda: input data (in matrix) projected onto covariance matrix eigenvectors (PCs)
            matrix_lda_fit: fit used to transform new data into reduced dimensional space
            lda.coef_: weight vectors
            lda.means_: means of each class of the original dataset
            lda.explained_variance_ratio_: amount of variance described by each PC
    """

    matrix = pd.DataFrame(matrix)

    lda = discriminant_analysis.LinearDiscriminantAnalysis()

    matrix_lda_fit = lda.fit(matrix, data_labels)
    matrix_lda = lda.transform(matrix)

    return matrix_lda, matrix_lda_fit, lda.coef_, lda.means_, lda.explained_variance_ratio_


def calc_mean_distance_vector(d2_matrix):

    return np.mean(d2_matrix, axis=0)


def filter_important_distances(upper_tri_d2_matrices, num_dists=75000):

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

    # print(selected_dist_atom_indexes)

    return important_distances_matrix, selected_dist_atom_indexes


def calc_num_atoms(vec_length):
    """
    Calculates number of atoms in a system based on the length of the vector generated by flattening the upper triangle
    of its interatomic distance matrix.
    :param vec_length: length of interatomic distance matrix vector
    :return: num_atoms: int, number of atoms in the system
    """

    n = Symbol('n', positive=True)
    answers = solve(n * (n - 1) / 2 - vec_length, n)
    num_atoms = int(answers[0])

    return num_atoms


def set_unimportant_distance_weights_to_zero(components, selected_dist_atom_indexes, num_atoms):

    num_dists = int((num_atoms*(num_atoms-1))/2)
    num_points = components.shape[0]

    components_all_distances = np.zeros((num_points, num_dists))

    distance_vector_indexes = list(pd.DataFrame(list(selected_dist_atom_indexes.values()))[1])
    for i in range(len(distance_vector_indexes)):
        components_all_distances[:, distance_vector_indexes[i]] = components[:, i]

    return components_all_distances


def generate_PC_matrices_selected_distances(n_dim, matrix_reduced, components, mean, selected_dist_atom_indexes,
                                            num_atoms):

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


def get_normal_modes_from_gaussian_output(path_to_log_file):

    normal_modes = []

    return normal_modes


def change_basis_to_normal_modes(normal_modes, components):
    """
    Change the basis of the matrix of principal components to be in terms of normal modes (which are linear combinations
    of Cartesian coordinates) rather than Cartesian coordinates. Does not work with an internal distances representation
    of molecular structures
    :param normal_modes: array of coefficients on Cartesian coordinates in normal modes of the molecule
    :param components: array of coefficients on Cartesian coordinates in Principal Components from PCA
    :return: components_normal_mode_basis: Principal Components from PCA expressed in a normal mode basis
    """

    components_normal_mode_basis = np.matmul(components, normal_modes)

    return components_normal_mode_basis


def inverse_transform_of_pcs(n_dim, matrix_reduced, components, mean):
    """
    Calculates the inverse transform of the PCs to see what the PCs correspond to in terms of geometric changes.
    Different than inverse_transform_of_pcs_as_normal_modes function because this function shows only structures that
    are spanned by the input data itself (i.e., uses the PC scores of each structure).
    :param n_dim: int, number of principal components specified to define reduced dimensional space
    :param matrix_reduced: array, PCs in reduced dimensional space
    :param components: array, eigenvectors of the covariance matrix of the input data
    :param mean: mean structure of the input data
    :return: PCs_separate, PCs_combined as arrays
    """
    PCs_separate = []
    for i in range(0, n_dim):
        PCi = np.dot(matrix_reduced[:, i, None], components[None, i, :]) + mean
        PCs_separate.append(PCi)

    PCs_combined = np.dot(matrix_reduced[:, 0:n_dim], components[0:n_dim, :]) + mean

    PCs_separate = np.array(PCs_separate)
    PCs_combined = np.array(PCs_combined)

    return PCs_separate, PCs_combined


def inverse_transform_of_pcs_as_normal_modes(n_dim, matrix_reduced, components, mean, alpha=40):
    """
    Adds incremental amounts of each eigenvector to the mean structure to show the effect of individual eigenvectors
    on molecular structure. Different than the inverse_transform_of_pcs function as this function does NOT take into
    account the space spanned by the original input data, but rather distorts the geometry of the mean structure in a
    linear fashion (i.e., how visualization of a normal mode appears in GaussView).
    :param n_dim: int, number of principal components specified to define reduced dimensional space
    :param components: array, eigenvectors of the covariance matrix of the input data
    :param mean: mean structure of the input data
    :param alpha: the multiple of each eigenvector to add to the mean structure
    :return: PCs_separate, PCs_combined as arrays
    """

    PCs_separate = []
    for i in range(0, n_dim):
        PCi = np.dot(alpha * (np.arange(-20, 21))[:, None], components[None, i, :]) + mean
        PCs_separate.append(PCi)

    multiplier = np.zeros((len(np.arange(-20, 21)), n_dim))
    for i in range(n_dim):
        multiplier[:, i] = np.arange(-20, 21)

    PCs_combined = np.dot(alpha * multiplier, components[0:n_dim, :]) + mean

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


def chirality_changes(reconstructed_coordinates, stereo_atoms, original_structure_signs):
    """ Determines chirality of structure along original trajectory and reconstructed reduced dimensional trajectory
     and switches inconsistencies along reduced dimensional IRC/trajectory.
    :param reconstructed_coordinates: coordinates of trajectory in the reduced dimensional space
    :param stereo_atoms: list of 4 indexes of atoms surrounding stereogenic center
    :param original_structure_signs: signs (positive or negative) that represent chirality at given point along original
    trajectory, numpy array
    :return: correct_chirality_coordinates: coordinates with the chirality of each structure consistent with the
    original coordinates (based on original_structure_signs), array
    """

    pos, neg, zero, signs_reconstructed = chirality_test(reconstructed_coordinates, stereo_atoms)
    correct_chirality_coordinates = reconstructed_coordinates

    for i in range(len(original_structure_signs)):
        if original_structure_signs[i] == 0:
            # If molecule begins planar but reconstruction of PCs are not, keep chirality consistent along reconstructed
            # trajectory
            if i > 0 and signs_reconstructed[i] != signs_reconstructed[0]:
                correct_chirality_coordinates[i] = -correct_chirality_coordinates[i]
        elif signs_reconstructed[i] != original_structure_signs[i]:
            correct_chirality_coordinates[i] = -correct_chirality_coordinates[i]

    return correct_chirality_coordinates


def chirality_changes_normal_modes(reconstructed_coordinates, stereo_atoms, original_structure_signs):
    """ Determines chirality of structure along original trajectory and reconstructed reduced dimensional trajectory
     and switches inconsistencies along reduced dimensional IRC/trajectory.
    :param reconstructed_coordinates: coordinates of trajectory in the reduced dimensional space
    :param stereo_atoms: list of 4 indexes of atoms surrounding stereogenic center
    :param original_structure_signs: signs (positive or negative) that represent chirality at given point along original
    trajectory, numpy array
    :return: correct_chirality_coordinates: coordinates with the chirality of each structure consistent with the
    original coordinates (based on original_structure_signs), array
    """

    pos, neg, zero, signs_reconstructed = chirality_test(reconstructed_coordinates, stereo_atoms)
    correct_chirality_coordinates = reconstructed_coordinates

    for i in range(len(reconstructed_coordinates)):
        if original_structure_signs[0] == 0:
            if i > 0 and signs_reconstructed[i] != signs_reconstructed[0]:
                correct_chirality_coordinates[i] = -correct_chirality_coordinates[i]
        elif signs_reconstructed[i] != original_structure_signs[0]:
            correct_chirality_coordinates[i] = -correct_chirality_coordinates[i]

    return correct_chirality_coordinates


def make_pc_xyz_files(output_directory, title, atoms, coordinates):
    """ Save principal coordinates as xyz files PC[n].xyz to output directory.
    :param output_directory: output directory to store xyz files, str
    :param atoms: atoms in input trajectory, list
    :param title: name of the input system, str
    :param coordinates: xyz coordinates of structures along PCi, list or numpy array
    :return: None
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


def print_prop_of_var_to_txt(values, system_name, directory):
    """
    Print list of proportions of variance explained by each principal component to a text file.
    :param values: array or list, proportions of variance in descending order
    :param system_name: name of the system, used for the text file name
    :param directory: output directory to put the output text file
    :return: None
    """
    normalized_values = values / np.sum(values)
    df = pd.DataFrame({'Principal Component': pd.Series([i+1 for i in range(len(values))]),
                       'Singular Value': values,
                       'Prop. of Variance': normalized_values,
                       'Cumul. Prop. of Var.': np.cumsum(normalized_values)})

    pd.set_option('display.expand_frame_repr', False)
    print(df.head())
    df.to_csv(os.path.join(directory, system_name + '_prop_of_var.txt'), sep='\t', index=None)


def print_distance_weights_to_files(directory, n_dim, system_name, pca_components, num_atoms, selected_atom_indexes=None):

    for n in range(n_dim):

        if selected_atom_indexes:
            distance_vector_indexes = list(pd.DataFrame(list(selected_atom_indexes.values()))[1])
        else:
            distance_vector_indexes = range(len(pca_components[n]))

        d = []
        for k, l in zip(distance_vector_indexes, range(len(pca_components[n]))):
            i, j = calc_ij(k, num_atoms)
            coeff = pca_components[n][l]
            d.append({'atom 1': i, 'atom 2': j, 'Coefficient of Distance': coeff})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(os.path.join(directory, system_name + '_PC%s_components.txt' % (n + 1)), sep='\t', index=None)


def print_distance_weights_to_files_select_atom_indexes(atom_indexes, n_dim, pca_components, system_name, directory):

    for n in range(n_dim):
        d = []
        for k in range(len(pca_components[n])):
            coeff = pca_components[n][k]
            d.append({'atom 1': atom_indexes[k][0], 'atom 2': atom_indexes[k][1], 'Coefficient of Distance': coeff})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(os.path.join(directory, system_name + '_PC%s_components.txt' % (n + 1)), sep='\t', index=None)


def print_distance_weights_to_files_weighted(directory, n_dim, system_name, pca_components, pca_values, num_atoms,
                                             display=False):

    for n in range(n_dim):
        d = []
        for k in range(len(pca_components[n])):
            i, j = calc_ij(k, num_atoms)
            coeff = (pca_values[n]/sum(pca_values))*pca_components[n][k]
            d.append({'atom 1': i, 'atom 2': j, 'Coefficient of Distance': coeff})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(os.path.join(directory, system_name + '_PC%s_components_weighted.txt' % (n + 1)), sep='\t',
                        index=None)

        if display:
            print("PC%s" % (n+1))
            print(sorted_d)


def transform_new_data(new_xyz_file_path, output_directory, n_dim, pca_fit, pca_components, pca_mean,
                       original_traj_coords, input_type, stereo_atoms=[1, 2, 3, 4], mw=False, remove_atom_types=None,
                       selected_atom_indexes=None):
    if input_type=="Cartesians":
        new_system_name, components_df = transform_new_data_cartesians(new_xyz_file_path, output_directory, n_dim,
                                                                       pca_fit, pca_components, pca_mean,
                                                                       original_traj_coords, mw=mw,
                                                                       remove_atom_types=remove_atom_types)
    elif input_type=="Distances":
        if selected_atom_indexes:
            new_system_name, components_df = transform_new_data_only_top_distances(new_xyz_file_path, output_directory, n_dim,
                                                                        pca_fit, pca_components, pca_mean,
                                                                        selected_atom_indexes=selected_atom_indexes,
                                                                        stereo_atoms=stereo_atoms, mw=mw,
                                                                        remove_atom_types=remove_atom_types)
        else:
            new_system_name, components_df = transform_new_data_distances(new_xyz_file_path, output_directory, n_dim,
                                                                      pca_fit, pca_components, pca_mean,
                                                                      stereo_atoms=stereo_atoms, mw=mw,
                                                                      remove_atom_types=remove_atom_types)
    else:
        print("ERROR: Please specify input_type=\"Cartesians\" or \"Distances\"")

    return new_system_name, components_df


def transform_new_data_cartesians(new_xyz_file_path, output_directory, n_dim, pca_fit, pca_components, pca_mean,
                                  original_traj_coords, mw=False, remove_atom_types=None):
    """
    Takes as input a new trajectory (xyz file) for a given system for which dimensionality reduction has already been
    conducted and transforms this new data into the reduced dimensional space. Generates a plot, with the new data atop
    the "trained" data, and generates xyz files for the new trajectories represented by the principal components.
    :param new_xyz_file_path: new input to dimensionality reduction (xyz file location), str
    :param output_directory: output directory, str
    :param n_dim: number of dimensions of the reduced dimensional space, int
    :param pca_fit: fit from PCA on training data
    :param pca_components: components from PCA on training data, array
    :param pca_mean: mean of input data to PCA (mean structure as coords or distances), array
    :param original_traj_coords: coordinates of the trajectory that the reduced dimensional space was trained on
    :param MW: whether coordinates should be mass weighted prior to PCA, bool
    """

    print("\nTransforming %s into reduced dimensional representation..." % new_xyz_file_path)

    new_system_name, atoms, coordinates = determine_type_and_read_file(new_xyz_file_path)

    if remove_atom_types is not None:
        atoms, coordinates = remove_atoms_by_type(remove_atom_types, atoms, coordinates)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("\nResults for %s input will be stored in %s" % (new_xyz_file_path, output_directory))

    # Determining names of output directories/files
    file_name_end = "_Cartesians"

    # Align structures using Kabsch algorithm so rotations don't affect PCs
    aligned_original_traj_coords = kabsch(original_traj_coords)
    coords_for_analysis = align_to_original_traj(coordinates, aligned_original_traj_coords)

    if mw is True:
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

    if mw is True:
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


def transform_new_data_distances(new_xyz_file_path, output_directory, n_dim, pca_fit, pca_components, pca_mean,
                                 stereo_atoms=[1, 2, 3, 4], mw=False, remove_atom_types=None):
    """
    Takes as input a new trajectory (xyz file) for a given system for which dimensionality reduction has already been
    conducted and transforms this new data into the reduced dimensional space. Generates a plot, with the new data atop
    the "trained" data, and generates xyz files for the new trajectories represented by the principal components.
    :param new_xyz_file_path: new input to dimensionality reduction (xyz file location), str
    :param output_directory: output directory, str
    :param n_dim: number of dimensions of the reduced dimensional space, int
    :param pca_fit: fit from PCA on training data
    :param pca_components: components from PCA on training data, array
    :param pca_mean: mean of input data to PCA (mean structure as coords or distances), array
    :param original_traj_coords: coordinates of the trajectory that the reduced dimensional space was trained on
    :param stereo_atoms: indexes of 4 atoms surrounding stereogenic center, list of ints
    :param MW: whether coordinates should be mass weighted prior to PCA, bool
    """

    print("\nTransforming %s into reduced dimensional representation..." % new_xyz_file_path)

    new_system_name, atoms, coordinates = determine_type_and_read_file(new_xyz_file_path)

    if remove_atom_types is not None:
        atoms, coordinates = remove_atoms_by_type(remove_atom_types, atoms, coordinates)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("\nResults for %s input will be stored in %s" % (new_xyz_file_path, output_directory))

    # Determining names of output directories/files
    file_name_end = "_Distances"

    if mw is True:
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

    if mw is True:
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


def transform_new_data_only_top_distances(new_xyz_file_path, output_directory, n_dim, pca_fit, pca_components, pca_mean,
                                 selected_atom_indexes, stereo_atoms=[1, 2, 3, 4], mw=False, remove_atom_types=None):
    """
    Takes as input a new trajectory (xyz file) for a given system for which dimensionality reduction has already been
    conducted and transforms this new data into the reduced dimensional space. Generates a plot, with the new data atop
    the "trained" data, and generates xyz files for the new trajectories represented by the principal components.
    :param new_xyz_file_path: new input to dimensionality reduction (xyz file location), str
    :param output_directory: output directory, str
    :param n_dim: number of dimensions of the reduced dimensional space, int
    :param pca_fit: fit from PCA on training data
    :param pca_components: components from PCA on training data, array
    :param pca_mean: mean of input data to PCA (mean structure as coords or distances), array
    :param original_traj_coords: coordinates of the trajectory that the reduced dimensional space was trained on
    :param stereo_atoms: indexes of 4 atoms surrounding stereogenic center, list of ints
    :param MW: whether coordinates should be mass weighted prior to PCA, bool
    """

    print("\nTransforming %s into reduced dimensional representation..." % new_xyz_file_path)

    new_system_name, atoms, coordinates = determine_type_and_read_file(new_xyz_file_path)

    if remove_atom_types is not None:
        atoms, coordinates = remove_atoms_by_type(remove_atom_types, atoms, coordinates)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("\nResults for %s input will be stored in %s" % (new_xyz_file_path, output_directory))

    if mw is True:
        coordinates_shifted = set_atom_one_to_origin(coordinates)
        mass_weighted_coords = mass_weighting(atoms, coordinates_shifted)
        coords_for_analysis = mass_weighted_coords

    else:
        coords_for_analysis = coordinates

    d2_vector_matrix_all = generate_and_reshape_ds_big_structures(coords_for_analysis)

    print('Starting new bit')
    num_dists = len(list(selected_atom_indexes.keys()))
    num_points = d2_vector_matrix_all.shape[0]

    important_distances_matrix = np.zeros((num_points, num_dists))
    distance_vector_indexes = list(pd.DataFrame(list(selected_atom_indexes.values()))[1])

    for i in range(len(distance_vector_indexes)):
        important_distances_matrix[:, i] = d2_vector_matrix_all[:, distance_vector_indexes[i]]

    components = pca_fit.transform(important_distances_matrix)
    components_df = pd.DataFrame(components)

    return new_system_name, components_df


def pathreducer(xyz_file_path, n_dim, stereo_atoms=[1, 2, 3, 4], input_type="Cartesians", mw=False, reconstruct=True,
                normal_modes=False, remove_atom_types=None, num_dists=75000, return_covariance=False):
    """
    Workhorse function for doing dimensionality reduction on xyz files. Dimensionality reduction can be done on the
    structures represented as Cartesian coordinates (easy/faster) or the structures represented as distances matrices
    (slower, but potentially more useful for certain systems that vary non-linearly with respect to Cartesian space,
    e.g., torsions).
    :param xyz_file_path: xyz file or directory filled with xyz files that will be used to generate the reduced
    dimensional space, str
    :param n_dim: number of dimensions to reduce system to using PCA, int
    :param stereo_atoms: list of 4 atom indexes surrounding stereogenic center, ints
    :param input_type: input type to PCA, either "Cartesians" or "Distances", str
    :return: name, directory, pca, pca_fit, components, mean, values, lengths
    """

    # Make sure even large matrices are printed out in their entirety (for the generation of xyz files)
    np.set_printoptions(threshold=sys.maxsize)

    # Check if input is directory (containing input files) or a single input file itself
    assert os.path.isfile(xyz_file_path) or os.path.isdir(xyz_file_path), "No such file or directory."

    if os.path.isfile(xyz_file_path) is True:
        if input_type == "Cartesians":
            system_name, output_directory, pca, pca_fit, components, mean, values, aligned_coords, covariance_matrix = \
                pathreducer_cartesians_one_file(xyz_file_path, n_dim, mw=mw, reconstruct=reconstruct,
                                                normal_modes=normal_modes, remove_atom_types=remove_atom_types, return_covariance=return_covariance)
        elif input_type == "Distances":
            system_name, output_directory, pca, pca_fit, components, mean, values, aligned_coords, selected_dist_atom_indexes, covariance_matrix = \
                pathreducer_distances_one_file(xyz_file_path, n_dim, stereo_atoms=stereo_atoms, mw=mw,
                                               reconstruct=reconstruct, normal_modes=normal_modes,
                                              remove_atom_types=remove_atom_types,  num_dists=num_dists, return_covariance=return_covariance)
        lengths = aligned_coords.shape[0]

    elif os.path.isdir(xyz_file_path) is True:
        if input_type == "Cartesians":
            system_name, output_directory, pca, pca_fit, components, mean, values, lengths, aligned_coords, covariance_matrix = \
                pathreducer_cartesians_directory_of_files(xyz_file_path, n_dim, mw=mw, reconstruct=reconstruct,
                                                          normal_modes=normal_modes, remove_atom_types=remove_atom_types, return_covariance=return_covariance)
        elif input_type == "Distances":
            system_name, output_directory, pca, pca_fit, components, mean, values, lengths, aligned_coords = \
                pathreducer_distances_directory_of_files(xyz_file_path, n_dim, stereo_atoms=stereo_atoms, mw=mw,
                                                         reconstruct=reconstruct, normal_modes=normal_modes,
                                                         num_dists=num_dists, remove_atom_types=remove_atom_types, return_covariance=return_covariance)

    return system_name, output_directory, pca, pca_fit, components, mean, values, lengths, aligned_coords, \
           selected_dist_atom_indexes if 'selected_dist_atom_indexes' in locals() else None, covariance_matrix if 'covariance_matrix' in locals() else None


def pathreducer_cartesians_one_file(xyz_file_path, n_dim, mw=False, reconstruct=True, normal_modes=False,
                                    remove_atom_types=None, return_covariance=False):
    """
    Workhorse function for doing dimensionality reduction on xyz files. Dimensionality reduction can be done on the
    structures represented as Cartesian coordinates (easy/faster) or the structures represented as distances matrices
    (slower, but potentially more useful for certain systems that vary in non-linear ways, e.g., torsions).
    :param xyz_file_path: xyz file or directory filled with xyz files that will be used to generate the reduced
    dimensional space, str
    :param n_dim: number of dimensions to reduce system to using PCA, int
    :return: name, directory, pca, pca_fit, components, mean, values, lengths
    """

    # Make sure even large matrices are printed out in their entirety (for the generation of xyz files)
    np.set_printoptions(threshold=sys.maxsize)

    # Check if input is directory (containing input files) or a single input file itself
    assert os.path.isfile(xyz_file_path) or os.path.isdir(xyz_file_path), "No such file or directory."

    # Determining names of output directories/files
    file_name_end = "_Cartesians"

    if mw is True:
        file_name_end = file_name_end + "_MW"
    elif mw is False:
        file_name_end = file_name_end + "_noMW"

    print("\nInput is one file.")
    system_name, atoms, coordinates = determine_type_and_read_file(xyz_file_path)

    if remove_atom_types is not None:
        atoms, coordinates = remove_atoms_by_type(remove_atom_types, atoms, coordinates)

    # Creating a directory for output (if directory doesn't already exist)
    output_directory = system_name + file_name_end + "_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Results for %s input will be stored in %s" % (xyz_file_path, output_directory))

    aligned_coords = kabsch(coordinates)
    print("\n(1C) Done aligning structures using Kabsch algorithm")

    if mw is True:
        mass_weighted_coordinates = mass_weighting(atoms, aligned_coords)
        print("\n(MW) Done mass-weighting coordinates!")

        matrix_for_pca = np.reshape(mass_weighted_coordinates, (mass_weighted_coordinates.shape[0],
                                                               mass_weighted_coordinates.shape[1] *
                                                               mass_weighted_coordinates.shape[2]))
    else:
        matrix_for_pca = np.reshape(aligned_coords, (aligned_coords.shape[0], aligned_coords.shape[1] *
                                                     aligned_coords.shape[2]))

    # PCA
    cartesians_pca, cartesians_pca_fit, cartesians_components, cartesians_mean, cartesians_values, covariance_matrix = \
        pca_dr(matrix_for_pca, return_covariance=return_covariance)

    print("\n(2) Done with PCA of Cartesian coordinates!")

    if reconstruct:
        if normal_modes:
            function = inverse_transform_of_pcs_as_normal_modes
            file_name_end += "_normal_modes"
        else:
            function = inverse_transform_of_pcs
        PCs_separate, PCs_combined = function(n_dim, cartesians_pca, cartesians_components,
                                                              cartesians_mean)

        print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")

        # Reshape n x 3N x 1 arrays into n x N x 3 arrays
        PCs_separate = np.reshape(PCs_separate, (PCs_separate.shape[0], PCs_separate.shape[1],
                                                     int(PCs_separate.shape[2] / 3), 3))

        PCs_combined = np.reshape(PCs_combined, (1, PCs_combined.shape[0], int(PCs_combined.shape[1] / 3), 3))

        if mw is True:
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
           cartesians_values, aligned_coords, covariance_matrix if 'covariance_matrix' in locals() else None


def pathreducer_cartesians_directory_of_files(xyz_file_directory_path, n_dim, mw=False, reconstruct=True,
                                              normal_modes=False, remove_atom_types=None, return_covariance=False):
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
    np.set_printoptions(threshold=sys.maxsize)

    # Check if input is directory (containing input files) or a single input file itself
    assert os.path.isfile(xyz_file_directory_path) or os.path.isdir(xyz_file_directory_path), "No such file or " \
                                                                                              "directory."

    # Determining names of output directories/files
    file_name_end = "_Cartesians"

    if mw is True:
        file_name_end = file_name_end + "_MW"
    elif mw is False:
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
        name, atoms_one_file, coordinates = determine_type_and_read_file(xyz_file)

        if remove_atom_types is not None:
            atoms_one_file, coordinates = remove_atoms_by_type(remove_atom_types, atoms_one_file, coordinates)

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

    if mw is True:
        mass_weighted_coordinates = mass_weighting(atoms_one_file, aligned_coords)
        print("\n(MW) Done mass-weighting coordinates!")

        matrix_for_pca = np.reshape(mass_weighted_coordinates, (mass_weighted_coordinates.shape[0],
                                    mass_weighted_coordinates.shape[1] * mass_weighted_coordinates.shape[2]))

    else:
        matrix_for_pca = np.reshape(aligned_coords, (aligned_coords.shape[0],
                                                           aligned_coords.shape[1] * aligned_coords.shape[2]))

    # PCA
    cartesians_pca, cartesians_pca_fit, cartesians_components, cartesians_mean, cartesians_values, covariance_matrix = \
        pca_dr(matrix_for_pca, return_covariance=return_covariance)
    print("\n(2) Done with PCA of Cartesian coordinates!")

    if reconstruct:
        if normal_modes:
            function = inverse_transform_of_pcs_as_normal_modes
            file_name_end += "_normal_modes"
        else:
            function = inverse_transform_of_pcs
        PCs_separate, PCs_combined = function(n_dim, cartesians_pca, cartesians_components,
                                                              cartesians_mean)
        print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")

        # Reshape n x 3N x 1 arrays into n x N x 3 arrays
        PCs_separate = np.reshape(PCs_separate, (PCs_separate.shape[0], PCs_separate.shape[1],
                                                     int(PCs_separate.shape[2] / 3), 3))

        PCs_combined = np.reshape(PCs_combined, (1, PCs_combined.shape[0], int(PCs_combined.shape[1] / 3), 3))

        if mw is True:
            # Remove mass-weighting of coordinates, individual Xs
            no_mass_weighting_PCs_separate = [remove_mass_weighting(atoms_one_file, PCs_separate[i]) for i in
                                              range(n_dim)]

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


def pathreducer_distances_one_file(xyz_file_path, n_dim, stereo_atoms=[1, 2, 3, 4], mw=False,
                                   print_distance_coefficients=True, reconstruct=True, normal_modes=False,
                                   num_dists=75000, remove_atom_types=None, return_covariance=False):
    """
    Workhorse function for doing dimensionality reduction on xyz files. Dimensionality reduction can be done on the
    structures represented as Cartesian coordinates (easy/faster) or the structures represented as distances matrices
    (slower, but potentially more useful for certain systems that vary in non-linear ways, e.g., torsions).
    :param xyz_file_path: xyz file or directory filled with xyz files that will be used to generate the reduced
    dimensional space, str
    :param n_dim: number of dimensions to reduce system to using PCA, int
    :param stereo_atoms: list of 4 atom indexes surrounding stereogenic center, ints
    :return: name, directory, pca, pca_fit, components, mean, values, lengths
    """

    # Make sure even large matrices are printed out in their entirety (for the generation of xyz files)
    np.set_printoptions(threshold=sys.maxsize)

    # Check if input is directory (containing input files) or a single input file itself
    assert os.path.isfile(xyz_file_path) or os.path.isdir(xyz_file_path), "No such file or directory."

    # Determining names of output directories/files
    file_name_end = "_Distances"
    if mw is True:
        file_name_end = file_name_end + "_MW"
    elif mw is False:
        file_name_end = file_name_end + "_noMW"

    print("\nInput is one file.")
    name, atoms, coordinates = determine_type_and_read_file(xyz_file_path)

    if remove_atom_types is not None:
        atoms, coordinates = remove_atoms_by_type(remove_atom_types, atoms, coordinates)

    # Creating a directory for output (if directory doesn't already exist)
    output_directory = name + file_name_end + "_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Results for %s input will be stored in %s" % (xyz_file_path, output_directory))

    aligned_coordinates = kabsch(coordinates)
    negatives, positives, zeroes, all_signs = chirality_test(aligned_coordinates, stereo_atoms)

    if mw is True:
        coordinates_shifted = set_atom_one_to_origin(coordinates)
        mass_weighted_coordinates = mass_weighting(atoms, coordinates_shifted)
        coords_for_pca = mass_weighted_coordinates

        print("\n(MW) Done mass-weighting coordinates!")

    else:
        coords_for_pca = aligned_coordinates

    if coords_for_pca.shape[1] > 1000:
        # num_dists = 75000
        print("Big matrix. Using the top %s distances for PCA..." % num_dists)
        d2_vector_matrix_all = generate_and_reshape_ds_big_structures(coords_for_pca)
        d2_vector_matrix, selected_dist_atom_indexes = filter_important_distances(d2_vector_matrix_all, num_dists=num_dists)

    else:
        d2_full_matrices = generate_distance_matrices(coords_for_pca)
        d2_vector_matrix = reshape_ds(d2_full_matrices)

    print("\n(1D) Generation of distance matrices and reshaping upper triangles into vectors done!")

    # PCA on distance matrix
    d_pca, d_pca_fit, d_components, d_mean, d_values, covariance_matrix = pca_dr(d2_vector_matrix, return_covariance=return_covariance)
    print("\n(2) Done with PCA of structures as distance matrices!")

    if print_distance_coefficients:
        if coords_for_pca.shape[1] > 1000:
            print_distance_weights_to_files(output_directory, n_dim, name + file_name_end, d_components, len(atoms),
                                            selected_atom_indexes=selected_dist_atom_indexes)

        else:
            print_distance_weights_to_files(output_directory, n_dim, name + file_name_end, d_components, len(atoms))

    if reconstruct:
        if normal_modes:
            function = inverse_transform_of_pcs_as_normal_modes
            file_name_end += "_normal_modes"
        else:
            function = inverse_transform_of_pcs

        if coords_for_pca.shape[1] > 1000:
            d_components = set_unimportant_distance_weights_to_zero(d_components, selected_dist_atom_indexes, len(atoms))
            d_mean = calc_mean_distance_vector(d2_vector_matrix_all)

        PCs_separate_d, PCs_combined_d = function(n_dim, d_pca, d_components, d_mean)
        print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")

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

        if mw is True:
            # Remove mass-weighting of coordinates, individual PCs
            no_mass_weighting_PCs_separate = [remove_mass_weighting(atoms, PCs_separate[i])
                                              for i in range(n_dim)]
            no_mass_weighting_PCs_combined = remove_mass_weighting(atoms, PCs_combined)
            print("\n(UMW) Done removing mass-weighting!")

        else:
            no_mass_weighting_PCs_separate = PCs_separate
            no_mass_weighting_PCs_combined = PCs_combined

        if normal_modes:
            chirality_consistent_PCs_separate = [kabsch(chirality_changes_normal_modes(no_mass_weighting_PCs_separate[i], stereo_atoms,
                                                                                       all_signs)) for i in range(n_dim)]

            # Reorient coordinates so they are in a consistent coordinate system/chirality, all Xs combined into one array
            chirality_consistent_PCs_combined = kabsch(chirality_changes_normal_modes(no_mass_weighting_PCs_combined, stereo_atoms,
                                                                                      all_signs))
        else:
            chirality_consistent_PCs_separate = [kabsch(chirality_changes(no_mass_weighting_PCs_separate[i], stereo_atoms,
                                                               all_signs)) for i in range(n_dim)]

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

    return name, output_directory, d_pca, d_pca_fit, d_components, d_mean, d_values, aligned_coordinates, \
           selected_dist_atom_indexes if 'selected_dist_atom_indexes' in locals() else None, covariance_matrix if \
               'covariance_matrix' in locals() else None


def pathreducer_distances_directory_of_files(trajectory_file_directory_path, n_dim, stereo_atoms=[1, 2, 3, 4], mw=False,
                                             print_distance_coefficients=True, reconstruct=True, normal_modes=False,
                                             num_dists=75000, remove_atom_types=None, return_covariance=False):
    """
    Workhorse function for doing dimensionality reduction on xyz files. Dimensionality reduction can be done on the
    structures represented as Cartesian coordinates (easy/faster) or the structures represented as distances matrices
    (slower, but potentially more useful for certain systems that vary in non-linear ways, e.g., torsions).
    :param trajectory_file_directory_path: xyz file or directory filled with xyz files that will be used to generate the
    reduced dimensional space, str
    :param n_dim: number of dimensions to reduce system to using PCA, int
    :param stereo_atoms: list of 4 atom indexes surrounding stereogenic center, ints
    :return: name, directory, pca, pca_fit, components, mean, values, lengths
    """
    # Check if input is directory (containing input files) or a single input file itself
    assert os.path.isfile(trajectory_file_directory_path) or os.path.isdir(trajectory_file_directory_path), "No such file or " \
                                                                                              "directory."

    print("\nInput is a directory of files.")

    # Make sure even large matrices are printed out in their entirety (for the generation of xyz files)
    np.set_printoptions(threshold=sys.maxsize)

    # Determining names of output directories/files
    file_name_end = "_Distances"
    if mw is True:
        file_name_end = file_name_end + "_MW"
    elif mw is False:
        file_name_end = file_name_end + "_noMW"

    # path = os.path.dirname(trajectory_file_directory_path)
    system_name = os.path.basename(trajectory_file_directory_path)
    print("\nDoing dimensionality reduction on files in %s" % system_name)

    for fname in os.listdir(trajectory_file_directory_path):
        if fname.endswith('.xyz'):
            all_files = sorted(glob.glob(os.path.join(trajectory_file_directory_path, '*.xyz')))
        elif fname.endswith('.trj'):
            all_files = sorted(glob.glob(os.path.join(trajectory_file_directory_path, '*.trj')))

    names = []
    atoms = []
    file_lengths = []
    i = 0
    for one_file in all_files:
        i = i + 1
        name, atoms_one_file, coordinates = determine_type_and_read_file(one_file)

        if remove_atom_types is not None:
            atoms_one_file, coordinates = remove_atoms_by_type(remove_atom_types, atoms_one_file, coordinates)

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

    if mw is True:
        coordinates_shifted = set_atom_one_to_origin(coords_for_analysis)
        mass_weighted_coordinates = mass_weighting(atoms_one_file, coordinates_shifted)
        print("\n(MW) Done mass-weighting coordinates!")
        coords_for_pca = mass_weighted_coordinates

    else:
        coords_for_pca = coords_for_analysis

    if coords_for_pca.shape[1] > 1000:
        # num_dists = num_dists
        print("Big matrix. Using the top %s distances for PCA..." % num_dists)
        d2_vector_matrix_all = generate_and_reshape_ds_big_structures(coords_for_pca)

        d2_mean = calc_mean_distance_vector(d2_vector_matrix_all)
        d2_vector_matrix, selected_dist_atom_indexes = filter_important_distances(d2_vector_matrix_all, num_dists=num_dists)

    else:
        d2_full_matrices = generate_distance_matrices(coords_for_pca)
        d2_vector_matrix = reshape_ds(d2_full_matrices)

    print("\n(1D) Generation of distance matrices and reshaping upper triangles into vectors done!")

    # PCA on distance matrix
    d_pca, d_pca_fit, d_components, d_mean, d_values, covariance_matrix = pca_dr(d2_vector_matrix, return_covariance=return_covariance)

    print("\n(2) Done with PCA of structures as interatomic distance matrices!")

    if print_distance_coefficients:
        print_distance_weights_to_files(output_directory, n_dim, system_name + file_name_end, d_components,
                                        len(atoms_one_file))

    if reconstruct:
        if normal_modes:
            function = inverse_transform_of_pcs_as_normal_modes
            file_name_end += "_normal_modes"
        else:
            function = inverse_transform_of_pcs

        if coords_for_pca.shape[1] > 1000:
            d_components = set_unimportant_distance_weights_to_zero(d_components, selected_dist_atom_indexes, len(atoms))
            d_mean = calc_mean_distance_vector(d2_vector_matrix_all)

        PCs_separate_d, PCs_combined_d = function(n_dim, d_pca, d_components, d_mean)

        print("\n(3) Done transforming reduced dimensional representation of input into full dimensional space!")
        # Turning distance matrix representations of structures back into Cartesian coordinates
        PCs_separate = [[distance_matrix_to_coords(PCs_separate_d[i][k])
                               for k in range(PCs_separate_d.shape[1])] for i in range(PCs_separate_d.shape[0])]
        # Turning distance matrix representations of structures back into Cartesian coordinates (all chosen PCs combined
        # into one xyz file)
        PCs_combined = [distance_matrix_to_coords(PCs_combined_d[i]) for i in range(np.array(PCs_combined_d).shape[0])]

        PCs_separate = np.real(PCs_separate)
        PCs_combined = np.real(PCs_combined)

        print("\n(4D)-(6D) Done with converting distance matrices back to Cartesian coordinates!")

        if mw is True:
            # Remove mass-weighting of coordinates, individual PCs
            no_mass_weighting_PCs_separate = [remove_mass_weighting(atoms_one_file, PCs_separate[i])
                                              for i in range(n_dim)]
            no_mass_weighting_PCs_combined = remove_mass_weighting(atoms_one_file, PCs_combined)
            print("\n(UMW) Done removing mass-weighting!")

        else:
            no_mass_weighting_PCs_separate = PCs_separate
            no_mass_weighting_PCs_combined = PCs_combined

        if normal_modes:
            chirality_consistent_PCs_separate = [kabsch(chirality_changes_normal_modes(no_mass_weighting_PCs_separate[i], stereo_atoms,
                                                                                       all_signs)) for i in range(n_dim)]
            chirality_consistent_PCs_combined = kabsch(chirality_changes_normal_modes(no_mass_weighting_PCs_combined, stereo_atoms,
                                                                                      all_signs))
        else:
            chirality_consistent_PCs_separate = [chirality_changes(no_mass_weighting_PCs_separate[i], stereo_atoms,
                                                                   all_signs)
                                                 for i in range(n_dim)]
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

        for x in range(len(all_files)):
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
            pathreducer_cartesians_one_file(input_path, n_dim, mw=mw)
    elif os.path.isdir(input_path) and input_type == "Cartesians":
        system_name, output_directory, pca, pca_fit, components, mean, singular_values, traj_lengths, aligned_coords = \
            pathreducer_cartesians_directory_of_files(input_path, n_dim, mw=mw)
    elif os.path.isfile(input_path) and input_type == "Distances":
        system_name, output_directory, pca, pca_fit, components, mean, singular_values, aligned_coords = \
            pathreducer_distances_one_file(input_path, n_dim, stereo_atoms=stereo_atoms, mw=mw)
    elif os.path.isdir(input_path) and input_type == "Distances":
        system_name, output_directory, pca, pca_fit, components, mean, singular_values, traj_lengths, aligned_coords = \
            pathreducer_distances_directory_of_files(input_path, n_dim, stereo_atoms=stereo_atoms, mw=mw)
    else:
        print("Something went wrong.")

    pcs_df = pd.DataFrame(pca)
    if os.path.isdir(input_path):
        lengths = traj_lengths
    else:
        lengths = None

    plot_variance = input("\nWould you like a plot of the variance captured by each PC? (True or False)\n")
    if plot_variance == "True":
        plotting_functions.plot_prop_of_var(singular_values, system_name, output_directory)
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
            new_system_name, new_data_df = transform_new_data_cartesians(new_input, output_directory, n_dim, pca_fit,
                                                                         components, mean, aligned_coords, MW=mw)
        elif input_type == "Distances":
            new_system_name, new_data_df = transform_new_data_distances(new_input, output_directory, n_dim, pca_fit,
                                                                        components, mean, stereo_atoms=stereo_atoms,
                                                                        MW=mw)
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
