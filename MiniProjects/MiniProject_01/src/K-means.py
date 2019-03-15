import numpy as np
import matplotlib.pyplot as plt
import random as rd
from collections import defaultdict
import copy

# load mfeat-pix dataset
pixel_data = np.loadtxt('../DigitsBasicRoutines/mfeat-pix.txt', dtype=np.uint)

m = 200 # number of training examples
n = pixel_data.shape[1] # number of features, 240
K = 2 # set the number of clusters
no_iters = 10 # the number of iterations to be performed by K-means

#[a, b) is the interval over which we obtain our training points (b exclusive)
a = 1400
b = 1600

clusters = defaultdict(list)
# randomly assign training points to K sets over the 
# params a, b: the interval over which to obtain the training points
def initialise_clusters():
    for i in range(a, b): # selecting sevens as data set
        clusters[i % K].append(pixel_data[i])

def calculate_cb_vecs():
    # setup the codebook vectors
    cb_vectors = np.zeros([n * K]).reshape(K, n)
    # calculate codebook vectors
    for i in range(K):
        sum = np.zeros([n], dtype=np.uint).reshape(1, n)
        for vector in clusters[i]:
            sum += vector
        # diviide the sum of the vectors by the size of the cluster
        cb_vectors[i] = np.divide(sum, len(clusters[i]))
    return cb_vectors


# tempDist stores distance between training points and codebook vectors
tempDist = np.zeros([K]).reshape(K, 1)
# tempCluster stores previous cluster composition
tempCluster = defaultdict(list)
# mat will contain the cluster numbers to reassign each vector
mat = np.zeros([m]).reshape(m, 1)

# reassign training points to clusters according to distance from codebook vectors
j = 0
initialise_clusters()
for run in range(no_iters): # number of runs of K means algorithm
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
            '''
                mat[j][0] contains the minimum distance of the vector in the jth position
                in the cluster dictionary
            '''
            j += 1

    # reset cluster information
    clusters.clear()

    # reassign vectors to new clusters
    # Note: new clusters are allocated in order of membership occurrence
    while(j >= 1):
        for k in tempCluster:
            for idx in range(len(tempCluster[k])):
                clusters[mat[m - j][0]].append(tempCluster[k][idx])
                j -= 1

# update the codebook vectors at the end of the loop
cb_vectors = calculate_cb_vecs()

# draw codebook vector for specified cluster
plt.imshow(cb_vectors[1,:].reshape(16,15))
plt.show()