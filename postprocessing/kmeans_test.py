# Create 50 datapoints in two clusters a and b
import numpy as np
from scipy.cluster.vq import kmeans, whiten
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

pts = 50

rng = np.random.default_rng()

a = rng.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
b = rng.multivariate_normal([30, 10],

                            [[10, 2], [2, 1]],

                            size=pts)

features = np.concatenate((a, b))

# Whiten data
whitened = whiten(features)

n_clusters=2

kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=None, algorithm='elkan').fit(features)

# # Find 2 clusters in the data
# codebook, distortion = kmeans(whitened, 2)

# Plot whitened data and cluster centers in red

# plt.scatter(whitened[:,0], whitened[:,1], c=kmeans.labels_)
plt.scatter(features[:,0], features[:,1], c=kmeans.labels_)

for i in np.arange(n_clusters):
    plt.scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1], c='r')

# plt.scatter(whitened[:, 0], whitened[:, 1])
# # plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
plt.legend()
plt.show()