import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#  Generate synthetic 2D point data 
def make_blobs(n_per_cluster=150, noise_points=40):
    centers = np.array([
        [2.0, 2.0],
        [-2.0, 0.0],
        [2.0, -2.0],
    ])
    cov = np.array([[0.25, 0.0],[0.0, 0.25]])  # fairly tight clusters
    
    clusters = []
    labels = []
    for i, c in enumerate(centers):
        pts = np.random.multivariate_normal(c, cov, size=n_per_cluster)
        clusters.append(pts)
        labels.append(np.full(n_per_cluster, i, dtype=int))
    
    X = np.vstack(clusters)
    y_true = np.concatenate(labels)
    
    # Add some uniform noise points (to imitate LiDAR outliers)
    noise = np.random.uniform(low=-4.0, high=4.0, size=(noise_points, 2))
    X = np.vstack([X, noise])
    y_true = np.concatenate([y_true, np.full(noise_points, -1, dtype=int)])
    return X, y_true


#  K-means implementation 
def kmeans(X, K=3, max_iter=100, tol=1e-4):
    N, D = X.shape
    
    # Initialize centroids by sampling K distinct points
    idx = np.random.choice(N, K, replace=False)
    centroids = X[idx].copy()
    
    for it in range(max_iter):
        # Assign step
        # distances: (N, K)
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        
        # Update step
        new_centroids = centroids.copy()
        for k in range(K):
            pts = X[labels == k]
            if len(pts) > 0:
                new_centroids[k] = pts.mean(axis=0)
            else:
                # Handle empty cluster by re-seeding to a random point
                new_centroids[k] = X[np.random.choice(N)]
        
        # Check convergence (movement of centroids)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            return labels, centroids
    
    return labels, centroids



X, y_true = make_blobs(n_per_cluster=150, noise_points=40)
K = 3
labels, centroids = kmeans(X, K=K, max_iter=100, tol=1e-4)

#  Visualization 
# Raw points
plt.figure(figsize=(6, 6))
plt.title("Raw 2D Points (LiDAR-like)")
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Clustered result
plt.figure(figsize=(6, 6))
plt.title(f"K-means Clustering (K={K}")
# color by labels; this uses default colormap without specifying colors
plt.scatter(X[:, 0], X[:, 1], c=labels, s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, edgecolors="k")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

