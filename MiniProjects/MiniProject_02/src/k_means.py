from collections import defaultdict
import numpy as np
import copy

class KMeans:

    """
    A class which encapsulates several functionalities which provide the 
    K-Means clustering algorithm.
    """

    def __init__(self, K):
        """
        Default constructor.

        :param K: The number of clusters.
        :type K: int
        """
        self.K = K
        self.cb_vectors = None

    def alternating_bins_initialisation(self, pixel_data, a=None, b=None):
        """ 
        Initialise clusters by alternating the bins to which the 
        vectors are assigned.

        :param pixel_data:  The data that is divded into clusters.  
        :type pixel_data: list[list[int]]
        :param a:   The lower bound of the interval in the pixel_data. If its set to None, then the value is 0.
        :type a: int
        :param b:   The upper bound of the interval in pixel_data. If is None then it is set to the length of pixel_data.
        :type b: int
        :rtype: defaultdict
        """

        if not a or not b:
            a = 0
            b = len(pixel_data)

        clusters = defaultdict(list)
        for i in range(a, b): # selecting sevens as data set
            clusters[i % self.K].append(pixel_data[i])
        
        return clusters

    def calculate_cb_vecs(self, clusters):
        """ 
        Setup and calculate codebook vectors 
        
        :param clusters: The clusters in which the codebook vectors are calculated.
        :type clusters: defaultdict
        :rtype: array(array(int))
        """
        if not clusters or not clusters[0]:
            return None

        # :param:`n` is the dimension of the vectors
        n = len(clusters[0][0])
        # Initialize the codebook vectors to 0
        cb_vectors = np.zeros([n * self.K]).reshape(self.K, n)
        for i in range(self.K):
            sum = np.zeros([n], dtype=np.uint).reshape(1, n)
            for vector in clusters[i]:
                sum += vector
            # divide the sum of the vectors by the size of the cluster
            cb_vectors[i] = np.divide(sum, len(clusters[i]))
        return cb_vectors

    def fix(self, pixel_data):

        """
        Runs the K-means algorithm.

        :param pixel_data: A set of vectors of the data which is clustered.
        :type pixel_data: array(array(int))
        """

        # :param:`m` is the size of :param:`pixel_data`
        m = len(pixel_data)

        # tempDist stores distance between training points and codebook vectors
        tempDist = np.zeros([self.K]).reshape(self.K, 1)
        # tempCluster stores previous cluster composition
        tempCluster = defaultdict(list)
        # mat will contain the cluster numbers to reassign each vector
        mat = np.zeros([m]).reshape(m, 1)
        tempMat = np.ones([m]).reshape(m, 1)

        j = 0
        # initialise clusters
        clusters = self.alternating_bins_initialisation(pixel_data)
        while not np.array_equal(tempMat, mat): # algorithm runs until the sets do not change
            tempMat = copy.deepcopy(mat)
            # cacluate codebook vectors for each cluster
            cb_vectors = self.calculate_cb_vecs(clusters)
            # preserve cluster information
            tempCluster = copy.deepcopy(clusters)

            for key in clusters: # for each cluster
                for index in range(len(clusters[key])): # for the length of the cluster
                    vector = clusters[key][index]
                    for i in range(self.K):
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
        cb_vectors = self.calculate_cb_vecs(clusters)
        self.cb_vectors = copy.deepcopy(cb_vectors)
        return cb_vectors

    def get_cb_vectors(self):
        """
        Get the codebook vectors.

        :rtype: array(array(int))
        """
        return self.cb_vectors