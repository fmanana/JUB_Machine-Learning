import numpy as np
import matplotlib.pyplot as plt
import random as rd
from collections import defaultdict
import copy

# load mfeat-pix dataset
pixel_data = np.loadtxt('mfeat-pix.txt', dtype=np.uint)

m = 200 # number of training examples
n = pixel_data.shape[1] # number of features, 240
K = 2 # set the number of clusters
no_iters = 4 # the number of iterations to be performed by K-means

# randomly assign training points to K sets
clusters = defaultdict(list)
for i in range(1400, 1600): # selecting sevens as data set
    clusters[i % K].append(pixel_data[i - 1])

# setup the codebook vectors
cb_vectors = np.zeros([n * K]).reshape(K, n)

# calculate codebook vectors
for i in range(K):
    sum = np.zeros([n], dtype=np.uint).reshape(1, n)
    for vector in clusters[i]:
        sum += vector
    # codebook vectors are the average of each cluster
    cb_vectors[i] = np.divide(sum, 200 / K)
    
# tempDist stores distance between training points and codebook vectors
tempDist = np.zeros([K]).reshape(K, 1)
# tempCluster stores previous cluster composition
tempCluster = defaultdict(list)
print("cl: ", tempCluster)
# mat will contain the cluster numbers to reassign each vector
mat = np.zeros([m]).reshape(m, 1)

# reassign training points to clusters according to distance from codebook vectors
j = 0
for run in range(no_iters): # number of runs of K means algorithm
    tempCluster = copy.deepcopy(clusters) # preserve previous cluster config
    for key in clusters: # each cluster
        for index in range(len(clusters[key])): # the length of the cluster
            vector = clusters[key][index]
            for i in range(K):
                tempDist[i] = np.c_[np.linalg.norm(vector - cb_vectors[i])]
            # print("tempDist:\n", tempDist)
            # print("min: ", np.argmin(tempDist))
            mat[j] = np.c_[np.argmin(tempDist)]
            j += 1
        # reset tempDist
        tempDist = np.zeros([K]).reshape(K, 1)
        # print(clusters)
        # clearing clusters
        clusters[key].clear()
    
'''
    # try to rearrange training points according to cluster values in mat
    while(j >= 1):
        for k in tempCluster:
            for idx in range(len(tempCluster[k])):
                # reassign vectors to new clusters
                clusters[mat[m - j][0]] = tempCluster[k][idx]
                print("mat hey:", idx)
                j -= 1
    tempDist = np.zeros([K]).reshape(K, 1)
    mat = np.zeros([m]).reshape(m, 1)
    j = 0
    # print("mat:\n", mat)
'''



# plt.imshow(cb_vectors[1,:].reshape(16,15))
# plt.show()
