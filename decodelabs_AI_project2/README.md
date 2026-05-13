# 🌸 Iris Flower Classification — KNN Supervised Learning
### DecodeLabs AI Intern Training | Batch 2026 | Project 2

---

## What This Project Is

This is a machine learning classification model built on the Iris dataset. The goal is simple: given 4 measurements of a flower (sepal length, sepal width, petal length, petal width), the model predicts which of 3 species it belongs to — Setosa, Versicolor, or Virginica.

It sounds simple. It isn't. Building this properly taught me the difference between *running code* and actually *understanding* what each line is doing and why the order matters.

---

## What I Actually Did

This wasn't just copy-paste-and-run. I went through the full supervised learning pipeline manually, tested edge cases, and observed what breaks when you skip steps.

**What I experimented with:**
- Ran the model at different K values (K=1, K=3, K=5, K=7, K=15, K=30) and observed how accuracy and the confusion matrix changed at each extreme
- Visualised feature distributions **before** and **after** StandardScaler to understand what scaling actually does to the data
- Watched what happens when you apply `fit_transform` on test data instead of just `transform` — and why that's wrong
- Traced exactly how a single flower travels through the entire pipeline from raw number to final prediction

---

## The Dataset

| Property | Value |
|---|---|
| Source | `sklearn.datasets.load_iris` |
| Total Samples | 150 (perfectly balanced) |
| Classes | 3 — Setosa, Versicolor, Virginica |
| Features | 4 — Sepal Length, Sepal Width, Petal Length, Petal Width |
| Class Distribution | 50 samples per class |

---

## Pipeline Overview

```
Raw Data (150 flowers, 4 features each)
        │
        ▼
  Define X (features) and y (labels)
        │
        ▼
  Train-Test Split  →  80% train (120 samples) | 20% test (30 samples)
        │
        ▼
  StandardScaler  →  fit on X_train only, transform both X_train and X_test
        │
        ▼
  KNN Classifier  →  model.fit(X_train, y_train)
        │
        ▼
  model.predict(X_test)
        │
        ▼
  Evaluate  →  Accuracy | Confusion Matrix | Classification Report | F1 Score
```

---

## The Code

### 1. Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)
```

### 2. Load and Inspect the Data

```python
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(df.shape)           # (150, 5)
print(df.head())
print(df.describe())
print(df['species'].value_counts())  # 50 each — balanced
```

### 3. Separate Features and Labels

```python
X = iris.data    # Shape: (150, 4) — the 4 measurements
y = iris.target  # Shape: (150,)  — species labels: 0, 1, or 2
```

`X` must be defined before it can be passed into anything else. `y` too. This order is not optional.

### 4. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 20% held back for testing
    random_state=42,   # Makes the split reproducible
    stratify=y         # Keeps class proportions equal in both sets
)

# Result: X_train=(120,4), X_test=(30,4), y_train=(120,), y_test=(30,)
```

The test set is locked from this point. The model never sees it during training.

### 5. Feature Scaling

```python
scaler = StandardScaler()

# Fit ONLY on training data — learn mean and std from training set
X_train = scaler.fit_transform(X_train)

# Apply the SAME transformation to test data — never refit on test
X_test = scaler.transform(X_test)
```

**Why this matters:** KNN uses Euclidean distance to find nearest neighbours. Without scaling, features with larger numeric ranges dominate the distance calculation — making the model effectively ignore the smaller-range features. Scaling puts all 4 features on equal footing.

**Why `fit` only on training data:** If you fit the scaler on test data too, you're letting the model peek at test statistics during preprocessing. That's data leakage — your evaluation numbers look better than they actually are.

### 6. Visualising Before vs After Scaling

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before scaling
axes[0].scatter(iris.data[:, 0], iris.data[:, 2],
                c=iris.target, cmap='viridis', alpha=0.7)
axes[0].set_title('Before Scaling')
axes[0].set_xlabel('Sepal Length'); axes[0].set_ylabel('Petal Length')

# After scaling
axes[1].scatter(X_train[:, 0], X_train[:, 2],
                c=y_train, cmap='viridis', alpha=0.7)
axes[1].set_title('After Scaling (StandardScaler)')
axes[1].set_xlabel('Sepal Length (scaled)'); axes[1].set_ylabel('Petal Length (scaled)')

plt.tight_layout()
plt.show()
```

### 7. Build and Train the Model

```python
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

For KNN, "training" just means storing the training points in memory. The actual computation happens at prediction time — for each new point, it finds the K nearest training points and takes a majority vote.

### 8. Predict and Evaluate

```python
predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions,
                            target_names=iris.target_names))
```

### 9. Confusion Matrix

```python
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix — KNN Iris Classification')
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.tight_layout()
plt.show()
```

### 10. Finding Optimal K — The Elbow Method

```python
error_rates = []

for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

plt.figure(figsize=(10, 5))
plt.plot(range(1, 31), error_rates, marker='o', linewidth=2, color='navy')
plt.title('Error Rate vs K Value')
plt.xlabel('K — Number of Neighbours')
plt.ylabel('Error Rate')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**What I observed:** K=1 gives near-perfect training accuracy but is unstable — it's just memorising. As K increases, the model generalises better up to a point, then starts underfitting as it considers too many neighbours to make a precise decision.

---

## Key Concepts This Project Taught Me

**Supervised Learning** — you give the model labelled examples and it learns to map inputs to outputs. No rules written manually. The machine finds the pattern.

**Train-Test Split** — you cannot fairly evaluate a model on the same data it trained on. The test set simulates real-world unseen data.

**Data Leakage** — fitting preprocessing steps (like scaling) on test data before evaluation gives falsely optimistic results. In production, that model would underperform.

**The Accuracy Mirage** — a model can have 95% accuracy and still be useless if the data is imbalanced. F1 score and the confusion matrix tell the real story.

**K in KNN** — K=1 overfits, K=100 underfits. The elbow curve finds the sweet spot where the model generalises best.

---

## Stack

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

*DecodeLabs Industrial Training — Batch 2026*

