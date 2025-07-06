import pandas as pd
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Ambil argumen dari MLflow Project
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None
dataset_path = sys.argv[3] if len(sys.argv) > 3 else "data_preprocessing.csv"

# Load dataset
df = pd.read_csv(dataset_path)

# Pisahkan fitur dan target
X = df.drop(columns=["stroke"])
y = df["stroke"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Oversampling
smote = SMOTE(random_state=42)
X_train_os, y_train_os = smote.fit_resample(X_train, y_train)

# Set experiment
mlflow.set_experiment("Stroke Prediction")

# Model training & logging
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train_os, y_train_os)

# Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Logging manual
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.log_metric("f1_score", f1)

# Log model
mlflow.sklearn.log_model(model, "model")
