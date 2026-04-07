# v1_train.py  (Python 3.12 compatible)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_synthetic_transactions(n_samples: int = 50000) -> pd.DataFrame:
    customer_ids = np.random.randint(1, 5000, size=n_samples)
    amounts = np.random.exponential(scale=50, size=n_samples)
    countries = np.random.choice(["US", "GB", "IN", "DE", "FR"], size=n_samples)
    channels = np.random.choice(["POS", "ECOM", "ATM"], size=n_samples)
    hours = np.random.randint(0, 24, size=n_samples)

    fraud_prob = (
        (amounts > 200).astype(float)
        + (countries == "IN").astype(float)
        + ((hours < 6) | (hours > 22)).astype(float)
    ) / 3.0

    labels = (np.random.rand(n_samples) < fraud_prob * 0.7).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "amount": amounts,
        "country": countries,
        "channel": channels,
        "hour": hours,
        "label": labels,
    })
    return df

def preprocess(df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = df["label"]
    X = pd.get_dummies(X, columns=["country", "channel"], drop_first=True)
    return X, y

def main():
    df = generate_synthetic_transactions()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    joblib.dump(model, artifacts_dir / "fraud_model.joblib")
    joblib.dump(list(X.columns), artifacts_dir / "feature_columns.joblib")

    print("Model saved to artifacts/")

if __name__ == "__main__":
    main()