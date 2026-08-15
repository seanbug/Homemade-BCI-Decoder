import numpy as np
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from joblib import dump

from features import extract_features

dataPartitions = 5
outputtedModel = "mi_clf.joblib"

files = glob.glob("mi_dataset_*.npz")
assert len(files) > 0, "No dataset files found."

X_all, y_all = [], []

for f in files:
    d = np.load(f, allow_pickle=True)
    eeg = d["eeg"]
    labels = d["labels"]
    fs = int(d["fs"])
    
    feats = np.array([extract_features(tr, fs_in=fs)[0] for tr in eeg])

    X_all.append(feats)
    y_all.append(labels)

X = np.concatenate(X_all)
y = np.concatenate(y_all)
X = X.reshape(-1, 1)

print(f"Loaded {len(files)} sessions")
print(f"Dataset size: {len(X)} trials")
print(f"Class balance: MI={sum(y)}, REST={len(y)-sum(y)}")

clf = LogisticRegression(max_iter=2000)
skf = StratifiedKFold(n_splits=dataPartitions, shuffle=True, random_state=42)

accs = []
for train_idx, test_idx in skf.split(X, y):
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])
    accs.append(accuracy_score(y[test_idx], preds))

print("\nCross-validation accuracy:")
for i, a in enumerate(accs):
    print(f" Fold {i+1}: {a:.3f}")

print(f"Mean Accuracy: {np.mean(accs):.3f}")

clf.fit(X, y)
dump(clf, outputtedModel)

print(f"\nSaved trained classifier: {outputtedModel}")

# confusion matrix gen
preds = clf.predict(X)
cm = confusion_matrix(y, preds)
print("\nConfusion Matrix Output:")
print(cm)
