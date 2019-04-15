from collections import defaultdict
import numpy as np
import copy

def alternating_bins_initialisation(pixel_data, K=10, a=0, b=2000):
    """ 
    Initialise clusters by alternating the bins to which the 
    vectors are assigned

    :param pixel_data: The data that is divded into clusters  
    :type pixel_data: list[list[int]]
    :param K: The number of clusters
    :type K: int
    :param a: The lower bound of the interval in the pixel_data.
    :type a: int
    :param b: The upper bound of the interval in pixel_data.
    :type b: int
    :rtype: defaultdict
    """
    if a < 0 or b > len(pixel_data):
        return None

    clusters = defaultdict(list)
    for i in range(a, b): # selecting sevens as data set
        clusters[i % K].append(pixel_data[i])
    
    return clusters

def calculate_cb_vecs(clusters):
    """ 
    Setup and calculate codebook vectors 
    
    :param clusters: The clusters in which the codebook vectors are calculated.
    :type clusters: defaultdict
    """
    if not clusters or not clusters[0]:
        return None

    # :param:`K` is the number of clusters
    K = len(clusters)
    # :param:`n` is the dimension of the vectors
    n = len(clusters[0][0])
    # Initialize the codebook vectors to 0
    cb_vectors = np.zeros([n * K]).reshape(K, n)
    for i in range(K):
        sum = np.zeros([n], dtype=np.uint).reshape(1, n)
        for vector in clusters[i]:
            sum += vector
        # divide the sum of the vectors by the size of the cluster
        cb_vectors[i] = np.divide(sum, len(clusters[i]))
    return cb_vectors

def k_means_clustering(pixel_data, K=10, a=0, b=2000):

    # TODO: This method needs drastic changes

    """K-means clustering algorithm"""
    if(K < 0):
        return None

    # :param:`m` is the size of :param:`pixel_data`
    m = b-a
    # :param:`n` is the size of the vectors in :param:`pixel_data`
    n = pixel_data.shape[1]

    # tempDist stores distance between training points and codebook vectors
    tempDist = np.zeros([K]).reshape(K, 1)
    # tempCluster stores previous cluster composition
    tempCluster = defaultdict(list)
    # mat will contain the cluster numbers to reassign each vector
    mat = np.zeros([m]).reshape(m, 1)
    tempMat = np.ones([m]).reshape(m, 1)

    j = 0
    # initialise clusters
    clusters = alternating_bins_initialisation(pixel_data, K, a, b)
    while not np.array_equal(tempMat, mat): # algorithm runs until the sets do not change
        tempMat = copy.deepcopy(mat)
        # cacluate codebook vectors for each cluster
        cb_vectors = calculate_cb_vecs(clusters)
        # preserve cluster information
        tempCluster = copy.deepcopy(clusters)

        for key in clusters: # for each cluster
            for index in range(len(clusters[key])): # for the length of the cluster
                vector = clusters[key][index]
                for i in range(K):
                    # save distances to each codebook vector
                    tempDist[i] = np.c_[np.linalg.norm(vector - cb_vectors[i])]

                mat[j][0] = np.c_[np.argmin(tempDist)]
                # mat[j][0] contains the minimum distance of the vector in the jth position
                # in the cluster dictionary
                j += 1

        # reset cluster information
        clusters.clear()

        # reassign training points to clusters according to distance from codebook vectors
        # Note: new clusters are allocated in order of membership occurrence
        while(j >= 1):
            for k in tempCluster:
                for idx in range(len(tempCluster[k])):
                    clusters[mat[m - j][0]].append(tempCluster[k][idx])
                    j -= 1

    # update the codebook vectors at the end of the loop
    cb_vectors = calculate_cb_vecs(clusters)
    return cb_vectors