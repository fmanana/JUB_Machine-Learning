import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# load mfeat-pix dataset
pixel_data = np.loadtxt('../DigitsBasicRoutines/mfeat-pix.txt', dtype=np.uint)

class pca:
    """
    :param m: the number of principle components we wish to obtain
    :param data: the original data set
    """
    def __init__(self, m, data):
        self.m = m
        self.data = data
    
    """ computes the mean of the cloud data """
    def __mean(self):
        counter = 0
        mean = np.zeros([len(self.data[0])]).reshape(len(self.data[0]),)
        for point in self.data:
            mean += point
            counter += 1
        mean = mean / counter
        return mean

    """ returns a list object with the centered data cloud """
    def __centerData(self):
        mean = self.__mean()
        centeredData = np.zeros([len(self.data) * len(self.data[0])]).reshape(len(self.data), len(self.data[0]))

        for i in range(len(self.data)):
            centeredData[i] = (self.data[i] - mean)

        return centeredData

    def __cov(self):
        data = self.__centerData()
        n = len(self.data)

        covariance = np.dot(data, np.transpose(data)) / n

        return covariance

    def __svd(self):
        cov = self.__cov()
        u, s, vh = np.linalg.svd(cov)

        return u, s, vh

    """ returns the PCA componenents colomn wise """
    def getFeatures(self):
        u, s, vh = self.__svd()

        features = np.zeros(len(u) * self.m).reshape(len(u), self.m)
        for i in range(self.m):
            for j in range(len(u)):
                features[j][i] = u[j][i]

        return features