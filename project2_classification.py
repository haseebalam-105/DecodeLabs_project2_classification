# ============================================================
# PROJECT 2: Data Classification Using AI
# DecodeLabs Industrial Training Kit | Batch 2026
# ============================================================
# Dataset   : Iris Benchmark (150 samples, 3 classes, 4 features)
# Algorithm  : K-Nearest Neighbors (KNN) — k=5
# Pipeline   : Load → Scale → Split (80/20) → Train → Evaluate
# Metrics    : Confusion Matrix + F1 Score (weighted)
# ============================================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# ── STEP 1: INPUT — Load & Explore the Iris Benchmark ──────────────────────
print("=" * 60)
print("  DecodeLabs | Project 2: Data Classification Using AI")
print("=" * 60)

iris = load_iris()
X = iris.data       # Features: sepal length, sepal width, petal length, petal width
y = iris.target     # Labels : 0=Setosa, 1=Versicolor, 2=Virginica

print(f"\n📊 Dataset Overview")
print(f"   Samples    : {X.shape[0]}")
print(f"   Features   : {X.shape[1]}  {iris.feature_names}")
print(f"   Classes    : {len(iris.target_names)}  {list(iris.target_names)}")

# ── STEP 2: PROCESS — Gatekeeper Rule: Feature Scaling ─────────────────────
# KNN uses Euclidean distance — unscaled features bias toward larger-range ones.
# StandardScaler transforms data to mean=0, variance=1.

scaler = StandardScaler()

# ── STEP 3: PROCESS — Structural Integrity: The Split ──────────────────────
# Shuffle before splitting to remove order bias (Iris is sorted by class).
# 80% training, 20% testing. random_state ensures reproducibility.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, random_state=42
)

# Fit scaler ONLY on training data to prevent data leakage into test set
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)          # Use same scaler params

print(f"\n🔀 Train/Test Split")
print(f"   Training samples : {len(X_train)} (80%)")
print(f"   Testing  samples : {len(X_test)}  (20%)")

# ── STEP 4: PROCESS — The Algorithm: K-Nearest Neighbors ───────────────────
# Proximity Principle: "Similar things exist in close proximity."
# k=5 → majority vote among 5 nearest neighbours in feature space.

k = 5
model = KNeighborsClassifier(n_neighbors=k)

# FIT  → Memorize the training map (KNN is lazy; stores training data)
model.fit(X_train_scaled, y_train)

# PREDICT → Apply proximity logic on the test set
predictions = model.predict(X_test_scaled)

# ── STEP 5: OUTPUT — Validation Metrics ────────────────────────────────────
# Accuracy alone is misleading on imbalanced data ("Accuracy Mirage").
# We use Confusion Matrix + F1 Score (harmonic mean of precision & recall).

accuracy = accuracy_score(y_test, predictions)
f1       = f1_score(y_test, predictions, average="weighted")
cm       = confusion_matrix(y_test, predictions)

print(f"\n📈 Model Performance (k={k})")
print(f"   Accuracy : {accuracy * 100:.2f}%")
print(f"   F1 Score : {f1:.4f}  (weighted average across all classes)")

print(f"\n🔢 Confusion Matrix")
print(f"   Rows = Actual | Columns = Predicted")
header = "           " + "  ".join(f"{n:>12}" for n in iris.target_names)
print(header)
for i, row in enumerate(cm):
    print(f"   {iris.target_names[i]:>10} " + "  ".join(f"{v:>12}" for v in row))

print(f"\n📋 Full Classification Report")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# ── BONUS: Predict on a new custom sample ──────────────────────────────────
print("=" * 60)
print("🔮 Predict a New Sample")
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])       # Likely Setosa
new_scaled  = scaler.transform(new_sample)
prediction  = model.predict(new_scaled)
proba       = model.predict_proba(new_scaled)[0]

print(f"   Input Features : {new_sample[0]}")
print(f"   Predicted Class: {iris.target_names[prediction[0]].upper()}")
print(f"   Confidence     : {max(proba)*100:.1f}%")
print(f"   Class Probs    : " + " | ".join(
    f"{iris.target_names[i]}: {p*100:.1f}%" for i, p in enumerate(proba)
))
print("=" * 60)
