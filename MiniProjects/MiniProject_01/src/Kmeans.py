import numpy as np
import matplotlib.pyplot as plt
import random as rd
from collections import defaultdict
import copy

# load mfeat-pix dataset
pixel_data = np.loadtxt('../DigitsBasicRoutines/mfeat-pix.txt', dtype=np.uint)

#[a, b) is the interval over which we obtain our training points (b exclusive)
a = 200
b = 400
m = b - a
"""m is the size of the set of our training points"""
n = pixel_data.shape[1]
"""n is the number of features"""
K = 200
"""K is the number of clusters. K=200 by default, can be changed"""

clusters = defaultdict(list)
# initialise clusters by alternating the bins to which the vectors are assigned
def alternating_bins_initialisation():
    """ 
    Initialise clusters by alternating the bins to which the 
    vectors are assigned
    """
    for i in range(a, b): # selecting sevens as data set
        clusters[i % K].append(pixel_data[i])

def in_order_initialisation():
    """ Assign the first m/K vectors to the first cluster and so on """
    i = 0
    for k in range(K):
        while(len(clusters[k]) < m/K and i < m):
            clusters[k].append(pixel_data[a + i])
            i += 1

def unbalanced_initialisation(offset):

    """
    Unevenly distributes vectors into clusters by placing offset 
    number of vectors in the first cluster and evenly spreads the 
    remaining vectors in the remaining clusters 

    :param offset: the first offset vectors are put in the first cluster  
    :type offset: int
    """

    if(K > 1):
        for i in range(a, a + offset):
            clusters[0].append(pixel_data[i])
        # the remaining vectors are spread evenly in the remaining clusters
        j = a + offset
        for k in range(1, K):
            while(len(clusters[k]) < (m - offset)/(K-1) and j < b):
                clusters[k].append(pixel_data[j])
                j += 1
    else:
        print("cannot have unbalanced initialisation with one cluster")


def calculate_cb_vecs():
    """ Setup and calculate codebook vectors """
    cb_vectors = np.zeros([n * K]).reshape(K, n)
    for i in range(K):
        sum = np.zeros([n], dtype=np.uint).reshape(1, n)
        for vector in clusters[i]:
            sum += vector
        # diviide the sum of the vectors by the size of the cluster
        cb_vectors[i] = np.divide(sum, len(clusters[i]))
    return cb_vectors


def main():

    """K-means clustering algorithm"""

    global K
    tmp_K = int(input("Enter K: "))
    if tmp_K in [1, 2, 3, 200]:
        K = tmp_K
    else:
        print("Invalid value: running K=1")

    # tempDist stores distance between training points and codebook vectors
    tempDist = np.zeros([K]).reshape(K, 1)
    # tempCluster stores previous cluster composition
    tempCluster = defaultdict(list)
    # mat will contain the cluster numbers to reassign each vector
    mat = np.zeros([m]).reshape(m, 1)
    tempMat = np.ones([m]).reshape(m, 1)

    j = 0
    # initialise clusters
    alternating_bins_initialisation()
    # in_order_initialisation()
    # unbalanced_initialisation(100)
    while not np.array_equal(tempMat, mat): # algorithm runs until the sets do not change
        tempMat = copy.deepcopy(mat)
        # cacluate codebook vectors for each cluster
        cb_vectors = calculate_cb_vecs()
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
    cb_vectors = calculate_cb_vecs()

    # draw codebook vector for specified cluster
    cluster_idx = rd.randint(0, K-1)
    plt.imshow(cb_vectors[cluster_idx,:].reshape(16,15))
    plt.show()

if __name__ == "__main__":
    main()