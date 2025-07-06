import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Load data hasil preprocessing
df = pd.read_csv("data_preprocessing.csv")

# Pisahkan fitur dan target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Oversampling hanya pada data latih
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Stroke Prediction")

with mlflow.start_run():
    mlflow.autolog()

    # Model tanpa tuning
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    # Logging eksplisit model
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.iloc[:5])

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
