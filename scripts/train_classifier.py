import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DataLandmarks")
data_file = os.path.join(folder, "landmarks_dataset.json")
model_file = os.path.join(os.path.dirname(folder), os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sign_model.pkl"))

print("Training Landmark Classifier...")

if not os.path.exists(data_file):
    print(f"Dataset not found at {data_file}. Please run datacollection_landmarks.py first!")
    exit(1)

with open(data_file, 'r') as f:
    data = json.load(f)

if not data:
    print("Dataset is empty!")
    exit(1)

X = []
y = []

for item in data:
    vec = item["landmarks"]
    # Re-normalize just to be perfectly safe (L2 norm)
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    X.append(vec.tolist())
    y.append(item["label"])

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using RandomForest for extremely robust non-linear boundaries on landmarks
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {acc * 100:.2f}%")

with open(model_file, 'wb') as f:
    pickle.dump(clf, f)

print(f"Perfect! Model saved to {model_file}")
