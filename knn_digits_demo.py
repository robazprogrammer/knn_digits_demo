# AI 600 - Week 3
# K-Nearest Neighbors Live Demo
# Lecture dataset: scikit-learn Digits dataset

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# BLOCK 1: IMPORTS AND LOAD THE DATA
# ============================================================

digits = load_digits()

X = digits.data
y = digits.target

print("Dataset shape:", X.shape)
print("Image shape:", digits.images[0].shape)
print("Target names:", digits.target_names)
print("First target label:", y[0])


# ============================================================
# BLOCK 2: DISPLAY A SAMPLE DIGIT
# ============================================================

plt.imshow(digits.images[0], cmap="gray_r")
plt.title(f"Label: {digits.target[0]}")
plt.axis("off")
plt.show()


# ============================================================
# BLOCK 3: EUCLIDEAN DISTANCE
# ============================================================

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


distance = euclidean_distance(X[0], X[1])

print("\nDistance between first two digit images:", round(distance, 4))


# ============================================================
# BLOCK 4: TRAINING DATA AND TEST DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# BLOCK 5: FIRST KNN MODEL
# ============================================================

raw_model = KNeighborsClassifier(n_neighbors=5)
raw_model.fit(X_train, y_train)

raw_predictions = raw_model.predict(X_test)
raw_accuracy = accuracy_score(y_test, raw_predictions)

print("\nUnscaled KNN accuracy:", round(raw_accuracy, 4))


# ============================================================
# BLOCK 6: SCALING CHECK
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_model = KNeighborsClassifier(n_neighbors=5)
scaled_model.fit(X_train_scaled, y_train)

scaled_predictions = scaled_model.predict(X_test_scaled)
scaled_accuracy = accuracy_score(y_test, scaled_predictions)

print("Scaled KNN accuracy:", round(scaled_accuracy, 4))


# ============================================================
# BLOCK 7: COMPARE VALUES OF k
# ============================================================

print("\nAccuracy by k:")

for k in range(1, 16):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    print(f"k={k:2d} accuracy={accuracy:.4f}")


# ============================================================
# BLOCK 8: EVALUATE THE MODEL
# ============================================================

print("\nConfusion matrix for scaled k=5:")
print(confusion_matrix(y_test, scaled_predictions))

print("\nClassification report for scaled k=5:")
print(
    classification_report(
        y_test,
        scaled_predictions,
        target_names=[str(n) for n in digits.target_names]
    )
)


# ============================================================
# BLOCK 9: CLASSIFY ONE MYSTERY DIGIT
# ============================================================

mystery_index = 0

plt.imshow(X_test[mystery_index].reshape(8, 8), cmap="gray_r")
plt.title("Mystery digit")
plt.axis("off")
plt.show()

mystery_prediction = scaled_model.predict(
    X_test_scaled[[mystery_index]]
)

mystery_vote_share = scaled_model.predict_proba(
    X_test_scaled[[mystery_index]]
)

print("\nMystery digit prediction:", mystery_prediction[0])
print("Neighbor vote proportions:", mystery_vote_share[0])
print("Actual digit:", y_test[mystery_index])


# ============================================================
# BLOCK 10: INSPECT THE NEAREST NEIGHBORS
# ============================================================

neighbor_distances, neighbor_indices = scaled_model.kneighbors(
    X_test_scaled[[mystery_index]],
    n_neighbors=5
)

neighbor_labels = y_train[neighbor_indices[0]]

print("\nNearest neighbor labels:", neighbor_labels)
print("Nearest neighbor distances:", neighbor_distances[0])
print("Vote count:", Counter(neighbor_labels))
