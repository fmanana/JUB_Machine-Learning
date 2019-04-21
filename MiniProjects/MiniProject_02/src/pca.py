import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict

class pca:
    """
    :param m: the number of principle components we wish to obtain
    :param data: the original data set
    """
    def __init__(self, m, data):
        self.m = m
        self.data = data
        self.N = data.shape[0]
        self.centeredData = None
        self.features = None
    
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

        centeredData = np.transpose(centeredData)
        self.centeredData = centeredData
        return centeredData

    def __cov(self):
        data = self.__centerData()
        N = len(self.data)

        covariance = np.dot(data, np.transpose(data)) / N

        return covariance

    def svd(self):
        cov = self.__cov()
        u, s, vh = np.linalg.svd(cov)

        return u, s, vh

    def getFeatures(self):
        return self.features

    def fit(self):
        u, s, vh = self.svd()

        um = u[:, :self.m]
        features = np.zeros([self.m * self.N]).reshape(self.m, self.N)
        for i in range(self.N):
            features[:, i:i + 1] = np.dot(np.transpose(um), self.centeredData[:, i:i + 1])

        self.features = deepcopy(features)

        return features

if __name__ == "__main__":
    # load mfeat-pix dataset
    pixel_data = np.loadtxt('../DigitsBasicRoutines/mfeat-pix.txt', dtype=np.uint)
    pca = pca(2, pixel_data)
    print(pca.getFeatures())