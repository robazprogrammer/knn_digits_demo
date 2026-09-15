# K-Nearest Neighbors Digits Demo

This repository contains a simple, step-by-step demonstration of the **K-Nearest Neighbors (KNN)** classification algorithm using the handwritten digits dataset included with `scikit-learn`.

The goal of the project is to make KNN easy to understand by connecting the underlying idea of **similarity and distance** to a complete Python classification workflow.

## What the Demo Covers

The script walks through:

1. Loading the handwritten digits dataset
2. Viewing a sample digit image
3. Calculating Euclidean distance
4. Splitting the data into training and test sets
5. Training a KNN classifier
6. Comparing unscaled and standardized features
7. Comparing different values of `k`
8. Evaluating the model with:
   - Accuracy
   - Confusion matrix
   - Classification report
9. Classifying a single mystery digit
10. Inspecting the nearest neighbors behind the prediction

## Dataset

The project uses the built-in `load_digits()` dataset from `scikit-learn`.

The dataset contains:

- **1,797 handwritten digit images**
- Digits from **0 through 9**
- Each image is **8 × 8 pixels**
- Each image is represented by **64 numerical features**

No external CSV or dataset download is required.

## How KNN Works

K-Nearest Neighbors classifies a new observation by finding the `k` closest labeled observations and allowing them to vote on the predicted class.

In simple terms:

> Find the examples that look most similar to the new observation and use their labels to make the prediction.

This demonstration uses **Euclidean distance** to measure similarity.

## Example Results

Using a reproducible 80/20 train-test split:

- Training observations: **1,437**
- Testing observations: **360**
- Unscaled KNN accuracy with `k = 5`: approximately **98.33%**
- Standardized KNN accuracy with `k = 5`: approximately **96.39%**

The script also compares values of `k` from 1 through 15.

In the demonstrated split, `k = 13` produces approximately **96.94% accuracy** using the standardized features.

One important lesson from this example is that scaling does not always improve performance. The pixel features in the digits dataset already use the same numerical scale, so standardization is not necessarily beneficial.

## Mystery Digit Example

The script selects one image from the test data and asks the KNN model to classify it.

In the demonstrated run:

- Predicted digit: **5**
- Actual digit: **5**
- Five nearest-neighbor labels: **5, 5, 5, 5, 5**

This makes the KNN voting process directly visible.

## Requirements

This project requires Python and the following packages:

```text
numpy
matplotlib
scikit-learn
