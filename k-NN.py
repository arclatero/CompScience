# --- k-NN Implementation ---
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, x):
        # Compute Euclidean distances manually
        distances = []
        for i in range(len(self.X_train)):
            dist = sum((x[j] - self.X_train[i][j]) ** 2 for j in range(len(x))) ** 0.5
            distances.append((dist, self.y_train[i]))

        # Sort distances and get k nearest labels
        for i in range(len(distances) - 1):
            for j in range(i + 1, len(distances)):
                if distances[i][0] > distances[j][0]:  # Simple sorting
                    distances[i], distances[j] = distances[j], distances[i]

        k_nearest_labels = [distances[i][1] for i in range(self.k)]

        # Majority voting
        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Return the most common label
        max_label = max(label_counts, key=label_counts.get)
        return max_label


# Default dataset
X_train = [[2, 3], [1, 1], [5, 4], [6, 8], [7, 5], [3, 2]]
y_train = [0, 0, 1, 1, 1, 0]

knn = KNN(k=3)
knn.fit(X_train, y_train)

# Ask user to add new data and classify it
while True:
    add_data = input("Do you want to add new data for classification? (yes/no): ").strip().lower()
    if add_data == "no":
        break
    elif add_data == "yes":
        x1 = float(input("Enter first feature: "))
        x2 = float(input("Enter second feature: "))
        new_point = [x1, x2]

        # Predict the class of the new point
        predicted_class = knn.predict(new_point)
        print(f"New data {new_point} is classified as Class {predicted_class}")
    else:
        print("Invalid input, please enter 'yes' or 'no'.")