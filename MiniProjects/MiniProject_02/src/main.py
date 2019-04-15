import matplotlib.pyplot as plt
import random
import numpy as np
from k_means import k_means_clustering

if __name__ == "__main__":
    # load mfeat-pix dataset
    pixel_data = np.loadtxt('../DigitsBasicRoutines/mfeat-pix.txt', dtype=np.uint)

    #[a, b) is the interval over which we obtain our training points (b exclusive)
    a = 200
    b = 400
    K = 3

    # draw codebook vector for specified cluster
    cluster_idx = random.randint(0, K-1)
    cb_vectors = k_means_clustering(pixel_data, K, a, b)
    plt.imshow(cb_vectors[cluster_idx,:].reshape(16,15))
    plt.show()