# --- k-Means Clustering ---
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []

    def fit(self, X):
        # Initialize centroids (pick first k points)
        self.centroids = [X[i] for i in range(self.k)]

        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            clusters = {}
            for i in range(self.k):
                clusters[i] = []

            for point in X:
                closest_idx = self._closest_centroid(point)
                clusters[closest_idx].append(point)

            # Compute new centroids
            new_centroids = []
            for i in range(self.k):
                if clusters[i]:
                    new_centroids.append(self._compute_centroid(clusters[i]))
                else:
                    new_centroids.append(self.centroids[i])  # Keep previous centroid

            # Stop if centroids do not change
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids

    def _closest_centroid(self, x):
        min_dist = None
        min_idx = 0
        for i in range(len(self.centroids)):
            dist = sum((x[j] - self.centroids[i][j]) ** 2 for j in range(len(x))) ** 0.5
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_idx

    def _compute_centroid(self, points):
        num_points = len(points)
        num_features = len(points[0])
        centroid = []
        for j in range(num_features):
            sum_feature = sum(points[i][j] for i in range(num_points))
            centroid.append(sum_feature / num_points)
        return centroid

    def predict(self, x):
        return self._closest_centroid(x)


# Default dataset
X_clusters = [[2, 3], [1, 1], [5, 4], [6, 8], [7, 5], [3, 2], [8, 9], [10, 11], [12, 14], [13, 15]]

kmeans = KMeans(k=3)
kmeans.fit(X_clusters)

# Output final cluster centroids
print("k-Means Final Cluster Centroids:")
for i, centroid in enumerate(kmeans.centroids):
    print(f"Cluster {i + 1}: {centroid}")

# Ask user to add new data and assign clusters dynamically
while True:
    add_data = input("Do you want to add new data for clustering? (yes/no): ").strip().lower()
    if add_data == "no":
        break
    elif add_data == "yes":
        x1 = float(input("Enter first feature: "))
        x2 = float(input("Enter second feature: "))
        new_point = [x1, x2]

        # Assign the new point to a cluster
        assigned_cluster = kmeans.predict(new_point)
        print(f"New data {new_point} belongs to Cluster {assigned_cluster + 1}")
    else:
        print("Invalid input, please enter 'yes' or 'no'.")