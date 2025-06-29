import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Inisialisasi MLflow ke localhost
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set nama eksperimen baru agar terpisah dari eksperimen sebelumnya
mlflow.set_experiment("Telco-Customer-Churn-Tuning")

# Load data hasil preprocessing
df = pd.read_csv("data_automate_processing.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisi hyperparameter range
n_estimators_range = np.linspace(50, 200, 4, dtype=int)  # contoh: 50, 100, 150, 200
max_depth_range = np.linspace(5, 20, 4, dtype=int)        # contoh: 5, 10, 15, 20

best_acc = 0
best_params = {}

# Loop pencarian kombinasi terbaik
for n_est in n_estimators_range:
    for max_d in max_depth_range:
        with mlflow.start_run(run_name=f"Tuning_RF_n{n_est}_d{max_d}"):
            # Aktifkan autolog (optional, jika ingin otomatis log semua)
            mlflow.sklearn.autolog()

            # Train model
            model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
            model.fit(X_train, y_train)

            # Predict dan hitung akurasi
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Log akurasi manual (autolog sudah mencatat juga)
            mlflow.log_metric("manual_accuracy", acc)

            # Simpan model terbaik
            if acc > best_acc:
                best_acc = acc
                best_params = {"n_estimators": n_est, "max_depth": max_d}
                mlflow.sklearn.log_model(model, artifact_path="model")

print(f"✅ Hyperparameter terbaik: {best_params} dengan akurasi: {best_acc:.4f}")
