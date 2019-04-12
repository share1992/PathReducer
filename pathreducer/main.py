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
        return self._backtransform_features(x)


class DistancePCA(PCA):
    def __init__(self, n_components=None, memory="normal"):
        super(DistancePCA, self).__init__(n_components=n_components)
        self.memory = memory

    def _transform_features(self, x):
        if x.ndim != 3 or x.shape[1:] != self._original_shape[1:]:
            raise systemexit("Error: Unexpected shape of data (%s)" % str(x.shape))

        distance_matrix = euclidian_distances(x, self.memory)
        print(distance_matrix[0,:3,:3])
        mask_i, mask_j = np.mask_indices(x.shape[1], np.triu, 1)
        return distance_matrix[:,mask_i, mask_j]

    def _vector_to_matrix(self, x):
        n_atoms = self._original_shape[1]
        distance_matrix = np.zeros((x.shape[0], n_atoms, n_atoms))
        mask_i, mask_j = np.mask_indices(n_atoms, np.triu, 1)
        distance_matrix[:, mask_i, mask_j] = x
        distance_matrix[:, mask_j, mask_i] = x
        return distance_matrix

    #np.repeat(x, 100, axis=0)
    #np.tile(x, (100,1))
    # ones @ x
    def _backtransform_features(self, x):
        distance_matrix = self._vector_to_matrix(x)

        d = distance_matrix[0]

        d_one = np.reshape(d[:, 0], (d.shape[0], 1))

        print(np.matmul(np.ones((d.shape[0], 1)), d[:,:1].T))

        m = (-0.5) * (d - np.matmul(np.ones((d.shape[0], 1)), d[:,:1].T) - np.matmul(d_one,
                                                                                           np.ones((1, d.shape[0]))))

        print(m[:3,:3])

        print(m.shape)
        quit()

    #values, vectors = np.linalg.eig(m)

    #idx = values.argsort()[::-1]
    #values = values[idx]
    #vectors = vectors[:, idx]

    #assert np.allclose(np.dot(m, vectors), values * vectors)

    #coords = np.dot(vectors, np.diag(np.sqrt(values)))

    ## Only taking first three columns as Cartesian (xyz) coordinates
    #coords = np.asarray(coords[:, 0:3])

    #return coords
