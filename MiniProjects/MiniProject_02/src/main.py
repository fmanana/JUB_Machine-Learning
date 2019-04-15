import matplotlib.pyplot as plt
import random
import numpy as np
from k_means import KMeans

if __name__ == "__main__":
    # load mfeat-pix dataset
    pixel_data = np.loadtxt('../DigitsBasicRoutines/mfeat-pix.txt', dtype=np.uint)

    K = 10
    km = KMeans(K)
    print("Doing K-means clustering with K={}".format(K))
    cb_vectors = km.fix(pixel_data)
    print("Done!")

    # draw codebook vector for specified cluster
    while True:

        cluster_idx = input("Input a number: ")
        if not cluster_idx:
            break
        cluster_idx = int(cluster_idx)
        if cluster_idx < 0 or cluster_idx >= 10:
            break
        plt.imshow(cb_vectors[cluster_idx,:].reshape(16,15))
        plt.show()