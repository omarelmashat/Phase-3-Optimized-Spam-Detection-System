"""Train and save the spam detection models."""

import os
import joblib
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from features import fit_features, save_vectorizer

TRAIN_PATH = "data/train.csv"
MODELS_DIR = "models"



def load_train_data(path=TRAIN_PATH):
    df = pd.read_csv(path)
    return df["Message"], df["Label"]


def build_models():
    """Create the three classifiers."""
    return {
        "naive_bayes": MultinomialNB(),
        # Use more iterations for text data.
        "logistic_regression": LogisticRegression(max_iter=1000),
        "svm": CalibratedClassifierCV(LinearSVC(), cv=3),
    }


def train_all(X_train_features, y_train):
    trained = {}
    for name, model in build_models().items():
        print(f"Training {name}...")
        model.fit(X_train_features, y_train)
        trained[name] = model
    return trained


def save_models(trained_models, out_dir=MODELS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for name, model in trained_models.items():
        path = os.path.join(out_dir, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"Saved {name} to {path}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train_text, y_train = load_train_data()

    # Fit on training data only.
    vectorizer, X_train_features = fit_features(X_train_text)
    save_vectorizer(vectorizer)

    trained_models = train_all(X_train_features, y_train)
    save_models(trained_models)

    print("\nDone. models/ now contains:")
    for f in sorted(os.listdir(MODELS_DIR)):
        print(" -", f)


if __name__ == "__main__":
    main()
