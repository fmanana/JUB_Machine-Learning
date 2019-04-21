import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from pca import pca


def init():

    """ Used to initialize several preconditions in order to execute later algorithms.
    Divide the data into training and testing data. Initialize z_i indicator vectors for each
    feature vectors (feature vectors are extracted later in the code) """

    global pixel_data, train_data, test_data

    # Divide the test and training data into separate sets
    train_data_idx = 0
    test_data_idx = 0
    for i in range(0, len(pixel_data), 200):
        for j in range(i, i + 100):
            train_data[train_data_idx] = deepcopy(pixel_data[j])
            train_data_idx += 1
        for j in range(i + 100, i + 200):
            test_data[test_data_idx] = deepcopy(pixel_data[j])
            test_data_idx += 1

    z = np.zeros(len(train_data)*10).reshape(len(train_data), 10)
    class_num = 0
    for i in range(0, len(train_data), 100):
        for j in range(i, i+100):
            z[j][class_num] = 1
        class_num += 1


# load mfeat-pix dataset
print("Loading pixel data from file...")
pixel_data = np.loadtxt('../DigitsBasicRoutines/mfeat-pix.txt', dtype=np.uint)
print("Done!")

n = pixel_data.shape[1]

train_data = np.zeros([1000*n]).reshape(1000, n)
test_data = np.zeros([1000*n]).reshape(1000, n)

init()

# TODO: Write the cross validation here
pca = pca(3, train_data)
features = pca.fit()

