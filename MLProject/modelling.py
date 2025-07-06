import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load dataset dari URL atau path lokal
df = pd.read_csv("data_preprocessing.csv")

# Pisahkan fitur dan target
X = df.drop(columns=["stroke"])
y = df["stroke"]

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Oversampling dengan SMOTE
smote = SMOTE(random_state=42)
X_train_os, y_train_os = smote.fit_resample(X_train, y_train)

# Set experiment (tanpa tracking URI)
mlflow.set_experiment("Stroke Prediction")

with mlflow.start_run():
    # Inisialisasi dan training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_os, y_train_os)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Logging manual
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Simpan model
    mlflow.sklearn.log_model(model, "model")
