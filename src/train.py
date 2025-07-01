# src/train.py
import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score)

# Initialize MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit_risk_modeling")

def load_data():
    """Load processed data from Task 4 with path validation"""
    data_path = os.path.join("data", "processed", "data_with_target.csv")
    abs_path = os.path.abspath(data_path)
    
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Processed data not found at: {abs_path}\n"
            "Please ensure:\n"
            "1. You're running from project root\n"
            "2. Task 4 has been executed\n"
            "Run first: python src/target_engineering_t4.py"
        )
    return pd.read_csv(abs_path)

def preprocess_data(df):
    """Split features/target with validation"""
    required_columns = {"is_high_risk", "CustomerId"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    X = df.drop(columns=["is_high_risk", "CustomerId"])
    y = df["is_high_risk"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_models(X_train, y_train):
    """Train and evaluate models with clear naming"""
    models = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {"C": [0.1, 1, 10], "penalty": ["l2"]}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(),
            "params": {"n_estimators": [50, 100], "max_depth": [5, 10]}
        }
    }
    
    results = {}
    for name, config in models.items():
        with mlflow.start_run(run_name=name):
            gs = GridSearchCV(
                config["model"], 
                config["params"], 
                cv=5, 
                scoring="roc_auc"
            )
            gs.fit(X_train, y_train)
            
            # Improved model logging with clear names
            mlflow.sklearn.log_model(
                sk_model=gs.best_estimator_,
                artifact_path=f"model_{name}",
                registered_model_name=f"CreditRisk_{name}"
            )
            
            mlflow.log_params(gs.best_params_)
            results[name] = gs.best_estimator_
    return results

def evaluate_model(model, X_test, y_test):
    """Calculate and log metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    mlflow.log_metrics(metrics)
    return metrics

def register_best_model(models_metrics):
    """Register the best model based on roc_auc score in MLflow Model Registry"""
    # Find the model with highest ROC AUC score
    best_model_name = max(models_metrics.items(), key=lambda x: x[1]["roc_auc"])[0]
    
    # Initialize MLflow client
    client = mlflow.tracking.MlflowClient()
    
    # Get the latest version of the best model
    model_version = client.get_latest_versions(
        f"CreditRisk_{best_model_name}", 
        stages=["None"]
    )[0].version
    
    # Transition the best model to Production stage
    client.transition_model_version_stage(
        name=f"CreditRisk_{best_model_name}",
        version=model_version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"\nRegistered {best_model_name} as Production model in MLflow Model Registry")
    return best_model_name

if __name__ == "__main__":
    try:
        print("Loading data...")
        df = load_data()
        
        print("Preprocessing...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        print("Training models...")
        models = train_models(X_train, y_train)
        
        print("\nEvaluation Results:")
        models_metrics = {}
        for name, model in models.items():
            metrics = evaluate_model(model, X_test, y_test)
            models_metrics[name] = metrics
            print(f"\n{name} Metrics:")
            for k, v in metrics.items():
                print(f"{k:>10}: {v:.4f}")
        
        # Register the best performing model
        best_model = register_best_model(models_metrics)
        print(f"\nBest model selected: {best_model}")
                
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Debugging Tips:")
        print("- Run from project root: 'python src/train.py'")
        print("- Verify 'data/processed/data_with_target.csv' exists")
        raise