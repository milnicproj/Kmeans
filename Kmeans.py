import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import unittest

# Functions for custom K-Means implementation
def initialize_centroids(X, n_clusters):
    return X[np.random.choice(X.shape[0], n_clusters, replace=False), :]

def compute_distances(X, centroids):
    return np.sum((centroids[np.newaxis, :, :] - X[:, np.newaxis, :])**2, axis=-1)

def update_centroids(X, km, n_clusters, centroids):
    new_centroids = []
    for c_ in range(n_clusters):
        cluster_points = X[km == c_]
        if len(cluster_points) > 0:
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            new_centroids.append(centroids[c_])
    return np.array(new_centroids)

def compute_SSE(X, km, centroids, n_clusters):
    SSE = 0
    for c_ in range(n_clusters):
        cluster_points = X[km == c_]
        if len(cluster_points) > 0:
            SSE += np.sum((cluster_points - centroids[c_])**2)
    return SSE

# Custom K-Means Testing
def test_kmeans():
    class TestKMeans(unittest.TestCase):
        def setUp(self):
            self.X, _ = make_blobs(n_samples=250, n_features=3, centers=5, random_state=345)
            self.n_clusters = 5

        def test_initialize_centroids(self):
            centroids = initialize_centroids(self.X, self.n_clusters)
            self.assertEqual(centroids.shape, (self.n_clusters, 3))

        def test_compute_distances(self):
            centroids = initialize_centroids(self.X, self.n_clusters)
            distances = compute_distances(self.X, centroids)
            self.assertEqual(distances.shape, (self.X.shape[0], self.n_clusters))

        def test_update_centroids(self):
            centroids = initialize_centroids(self.X, self.n_clusters)
            km = np.random.randint(0, self.n_clusters, size=self.X.shape[0])
            new_centroids = update_centroids(self.X, km, self.n_clusters, centroids)
            self.assertEqual(new_centroids.shape, (self.n_clusters, 3))

        def test_compute_SSE(self):
            centroids = initialize_centroids(self.X, self.n_clusters)
            km = np.random.randint(0, self.n_clusters, size=self.X.shape[0])
            SSE = compute_SSE(self.X, km, centroids, self.n_clusters)
            self.assertIsInstance(SSE, float)

    unittest.main(argv=[''], exit=False)

# MNIST Data Processing
mnist = fetch_openml('mnist_784', version=1)
X_mnist = mnist.data.values
y_mnist = mnist.target.astype(int)
scaler = StandardScaler()
X_mnist_scaled = scaler.fit_transform(X_mnist)

# Custom K-Means on MNIST
n_clusters = 10
n_iterations = 15
centroids_custom = initialize_centroids(X_mnist_scaled, n_clusters)

for i in range(n_iterations):
    distances = compute_distances(X_mnist_scaled, centroids_custom)
    km_custom = np.argmin(distances, axis=1)
    centroids_custom = update_centroids(X_mnist_scaled, km_custom, n_clusters, centroids_custom)
    SSE_custom = compute_SSE(X_mnist_scaled, km_custom, centroids_custom, n_clusters)
    print(f"Iteration {i+1}, SSE: {SSE_custom}")

# Sklearn K-Means for Comparison
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_mnist_scaled)
labels_sklearn = kmeans.labels_
SSE_sklearn = kmeans.inertia_

# Comparison of SSE
print(f"SSE (Custom Implementation): {SSE_custom}")
print(f"SSE (Sklearn): {SSE_sklearn}")

# Confusion Matrices
conf_matrix_custom = confusion_matrix(y_mnist, km_custom)
conf_matrix_sklearn = confusion_matrix(y_mnist, labels_sklearn)

# Visualization of Confusion Matrices
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_custom, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Custom Implementation)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_sklearn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Sklearn)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.tight_layout()
plt.show()

# Display Random MNIST Image with Cluster Assignment
def show_random_image(X, y, km_labels, centroids):
    idx = np.random.randint(0, X.shape[0])
    image = X[idx].reshape(28, 28)

    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"True Label: {y[idx]}")
    plt.axis("off")

    # Cluster Centroid
    cluster = km_labels[idx]
    cluster_centroid = centroids[cluster].reshape(28, 28)

    plt.subplot(1, 2, 2)
    plt.imshow(cluster_centroid, cmap="gray")
    plt.title(f"Assigned Cluster: {cluster}")
    plt.axis("off")

    plt.show()

# Show a random image
def main():
    show_random_image(X_mnist_scaled, y_mnist, km_custom, centroids_custom)

if __name__ == "__main__":
    test_kmeans()
    main()
