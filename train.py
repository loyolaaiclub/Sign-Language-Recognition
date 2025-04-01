# train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

TRAIN_CSV = "sign_mnist_train.csv"
TEST_CSV = "sign_mnist_test.csv"
MODEL_FILE = "gesture_model.pkl"

print("DEBUG: Reading training data from", TRAIN_CSV)
train_df = pd.read_csv(TRAIN_CSV)
print("DEBUG: Training data shape:", train_df.shape)

print("DEBUG: Reading test data from", TEST_CSV)
test_df = pd.read_csv(TEST_CSV)
print("DEBUG: Test data shape:", test_df.shape)

# Separate features and target.
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# Normalize pixel values to the range [0, 1].
X_train = X_train / 255.0
X_test = X_test / 255.0

print("DEBUG: Training RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate on the test set.
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"DEBUG: Test Accuracy: {accuracy:.4f}")

# Save the trained model.
with open(MODEL_FILE, "wb") as f:
    pickle.dump(clf, f)

print(f"DEBUG: Model trained and saved to {MODEL_FILE}")
