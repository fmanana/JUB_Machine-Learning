import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from pca import pca


def init():

    """ Used to initialize several preconditions in order to execute later algorithms.
    Divide the data into training and testing data. Initialize z_i indicator vectors for each
    feature vectors (feature vectors are extracted later in the code) """

    global pixel_data, train_data, test_data, z

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
train = pca(34, train_data)
test = pca(34, test_data)
train_features = train.fit()
test_features = test.fit()


""" returns the decision function/matrix with optimal weights w_opt """
def linreg(features):
    phi = np.transpose(features)

    w_opt = np.dot(np.transpose(phi), z)
    temp = np.dot(np.transpose(phi), phi)
    w_opt = np.dot(np.linalg.inv(temp), w_opt)
    w_opt = np.transpose(w_opt)
    
    return w_opt

W = linreg(train_features)

results = np.dot(W, test_features)
results = np.transpose(results)


num = 0
cnt = 0
correct = 0
incorrect = 0
for vector in results:
    if(cnt >= 100):
        cnt = 0
        num += 1    
    guess = np.argmax(vector)
    if guess == num:
        correct += 1
    else:
        incorrect += 1
    cnt += 1

print(correct)
print(incorrect)