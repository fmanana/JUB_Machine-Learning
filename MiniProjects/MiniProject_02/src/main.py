# import matplotlib.pyplot as plt
# import numpy as np
# from k_means import KMeans
import matplotlib.pyplot as plt
import numpy as np
from k_means import KMeans

def tryout(cb_vectors):
    """ Just for checking the codebook vectors, ignore this function. """
    while True:
        cluster_idx = input("Input a number: ")
        if not cluster_idx:
            break
        cluster_idx = int(cluster_idx)
        if cluster_idx < 0 or cluster_idx >= 10:
            break
        plt.imshow(cb_vectors[cluster_idx,:].reshape(16,15))
        plt.show()


def init():

    """ Used to initialize several preconditions in order to execute later algorithms.
    Divide the data into training and testing data. Initialize z_i indicator vectors for each
    feature vectors (feature vectors are extracted later in the code) """

    global pixel_data, training_data, test_data

    # Divide the test and training data into separate sets
    for i in range(0, len(pixel_data), 200):
        for j in range(i, i + 100):
            training_data.append(pixel_data[j])
        for j in range(i + 100, i + 200):
            test_data.append(pixel_data[j])

    # TODO: Structure the code
    # TODO: Create target vectors z_i

    z = np.zeros(len(training_data)*10).reshape(len(training_data), 10)
    class_num = 0
    for i in range(0, len(training_data), 100):
        for j in range(i, i+100):
            z[j][class_num] = 1
        class_num += 1


# load mfeat-pix dataset
print("Loading pixel data from file...")
pixel_data = np.loadtxt('../DigitsBasicRoutines/mfeat-pix.txt', dtype=np.uint)
print("Done!")

training_data = []
test_data = []
K = 10

init()

km = KMeans(K)
print("Doing K-means clustering with K={}".format(K))
cb_vectors = km.fix(training_data)
print("Done!")
# tryout(cb_vectors)
features = km.extract_features(training_data)