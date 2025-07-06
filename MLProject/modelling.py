import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import sys

# Ambil path dataset dari argumen command-line (opsional)
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
else:
    dataset_path = "data_preprocessing.csv"  # default fallback

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

# Set experiment (hindari error: biarkan MLflow Project yang handle run-nya)
mlflow.set_experiment("Stroke Prediction")

# Jangan gunakan `with mlflow.start_run()` di MLflow Project!
# Logging secara manual
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_os, y_train_os)

y_pred = model.predict(X_test)

# Hitung metrik
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log param dan metrik
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.log_metric("f1_score", f1)

# Simpan model
mlflow.sklearn.log_model(model, "model")
