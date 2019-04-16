from sklearn.decomposition import PCA as sklPCA
import numpy as np

from .utils import euclidian_distances

class PCA(sklPCA):
    def __init__(self, n_components=None):
        self._model = sklPCA(n_components=n_components)
        self._original_shape = None

    def _transform_features(self, x):
        if x.ndim != 3 or x.shape[1:] != self._original_shape[1:]:
            raise systemexit("Error: Unexpected shape of data (%s)" % str(x.shape))
        return x.reshape(x.shape[0], -1)

    def _backtransform_features(self, x):
        return x.reshape(x.shape[0], *self._original_shape[1:])

    def fit(self, x):
        self._original_shape = x.shape
        x = self._transform_features(x)
        return self._model.fit(x)

    def transform(self, x):
        x = self._transform_features(x)
        return self._model.transform(x)

    def inverse_transform(self, x):
        if x.ndim != 2 or x.shape[-1] != self._model.n_components:
            raise systemexit("Error: Unexpected shape of data (%s)" % str(x.shape))
        x = self._model.inverse_transform(x)
        #x = np.dot(x[:,0,None], self._model.components_[None,0,:]) + self._model.mean_
        return self._backtransform_features(x)


class DistancePCA(PCA):
    def __init__(self, n_components=None, memory="normal"):
        super(DistancePCA, self).__init__(n_components=n_components)
        self.memory = memory

    def get_squared_euclidian_distances(self, x):
        # Faster than np.linalg.norm
        return np.sum((x[:,:,None] - x[:,None,:])**2, axis=3)

    def _transform_features(self, x):
        if x.ndim != 3 or x.shape[1:] != self._original_shape[1:]:
            raise systemexit("Error: Unexpected shape of data (%s)" % str(x.shape))

        distance_matrix = self.get_squared_euclidian_distances(x)
        mask_i, mask_j = np.mask_indices(x.shape[1], np.triu, 1)
        return distance_matrix[:,mask_i, mask_j]

    def _vector_to_matrix(self, x):
        n_atoms = self._original_shape[1]
        distance_matrix = np.zeros((x.shape[0], n_atoms, n_atoms))
        mask_i, mask_j = np.mask_indices(n_atoms, np.triu, 1)
        distance_matrix[:, mask_i, mask_j] = x
        distance_matrix[:, mask_j, mask_i] = x
        return distance_matrix

    def _backtransform_features(self, x):
        """
        Using the gram matrix approach of https://arxiv.org/abs/1502.07541
        """

        distance_matrix = self._vector_to_matrix(x)
        n = distance_matrix.shape[1]

        one_d = np.repeat(distance_matrix[:,:1], n, axis=1)
        d_one = np.transpose(one_d, axes=(0,2,1))
        g = - 0.5 * (distance_matrix - one_d - d_one)

        eigen_values, eigen_vectors = np.linalg.eig(g)
        idx = eigen_values.argsort(1)[:,::-1]
        # Convert to 3d dummy array to be able to vectorize the
        # take_along_axis operation on eigen_vectors
        idx_3d = np.repeat(idx[:,None], n, axis=1)
        sorted_eigen_values = np.take_along_axis(eigen_values, idx, axis=1)
        sorted_eigen_vectors = np.take_along_axis(eigen_vectors, idx_3d, axis=2)
        identity = np.repeat(np.identity(n)[None], distance_matrix.shape[0], axis=0)
        # clip(0) for removing negative eigen values to avoid nan's in the einsum
        coords = sorted_eigen_vectors @ np.einsum('ijk,ij->ijk', identity, np.sqrt(sorted_eigen_values.clip(0)))

        # Only first three values from the last dimension corresponds to coordinates
        return coords[:,:,:3]
