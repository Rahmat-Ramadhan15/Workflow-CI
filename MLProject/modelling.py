import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import sys
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Ambil parameter dari argumen CLI
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else "data_preprocessing.csv"

    # Load dataset
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Oversampling
    smote = SMOTE(random_state=42)
    X_train_os, y_train_os = smote.fit_resample(X_train, y_train)

    # Contoh input untuk input_example MLflow
    input_example = X_train_os.iloc[:5]

    # Mulai tracking MLflow
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train_os, y_train_os)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Logging ke MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"\nModel selesai dilatih.")
        print(f"Akurasi: {acc:.4f} | Presisi: {prec:.4f} | Recall: {rec:.4f} | F1-score: {f1:.4f}")
