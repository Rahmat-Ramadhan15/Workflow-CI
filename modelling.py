import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# =======================
# Argument dari MLproject
# =======================
n_estimators_default = int(sys.argv[1]) if len(sys.argv) > 1 else 200
max_depth_default = int(sys.argv[2]) if len(sys.argv) > 2 else 10
dataset_path = sys.argv[3] if len(sys.argv) > 3 else "data_automate_processing.csv"

# Inisialisasi experiment
mlflow.set_experiment("Telco-Customer-Churn-Tuning")

# Load dataset
df = pd.read_csv(dataset_path)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aktifkan autolog
mlflow.sklearn.autolog()

# Tandai nama run (karena tidak bisa pakai start_run saat mlflow run)
mlflow.set_tag("mlflow.runName", f"Tuning_RF_n{n_estimators_default}_d{max_depth_default}")

# Train model
model = RandomForestClassifier(n_estimators=n_estimators_default, max_depth=max_depth_default, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
mlflow.log_metric("manual_accuracy", acc)

# Simpan model
mlflow.sklearn.log_model(model, artifact_path="model")

print(f"✅ Model dilatih dengan n_estimators={n_estimators_default}, max_depth={max_depth_default}")
print(f"🎯 Akurasi: {acc:.4f}")
