import numpy as np

def euclidian_distances(x, memory="normal"):
    if memory == "normal":
        return normal_memory_euclidian_distances(x)
    elif memory == "low":
        return low_memory_euclidian_distances(x)
    elif memory == "verylow":
        return verylow_memory_euclidian_distances(x)

def normal_memory_euclidian_distances(x):
    return np.linalg.norm(x[:,:,None] - x[:,None,:], axis=3)

def low_memory_euclidian_distances(x):
    distances = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        distances[i] = np.linalg.norm(x[i,:,None] - x[i,None,:], axis=2)
    return distances

def verylow_memory_euclidian_distances(x):
    distances = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]-1):
            for k in range(j+1, x.shape[1]):
                distances[i,j,k] = np.linalg.norm(x[i,j] - x[i,k])
                distances[i,k,j] = distances[i,j,k]
    return distances
