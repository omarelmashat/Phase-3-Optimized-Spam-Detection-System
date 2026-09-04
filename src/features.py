"""Create and save TF-IDF features for the messages."""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZER_PATH = "models/vectorizer.pkl"


def build_vectorizer(max_features=5000, ngram_range=(1, 2)):
    """Create a new TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )


def fit_features(X_train, max_features=5000, ngram_range=(1, 2)):
    """Fit the vectorizer on training text only."""
    vectorizer = build_vectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_features = vectorizer.fit_transform(X_train)
    return vectorizer, X_train_features


def transform_features(vectorizer, X):
    """Transform text with a fitted vectorizer."""
    return vectorizer.transform(X)


def save_vectorizer(vectorizer, path=VECTORIZER_PATH):
    joblib.dump(vectorizer, path)
    print(f"Saved fitted vectorizer to {path}")


def load_vectorizer(path=VECTORIZER_PATH):
    """Load a saved vectorizer."""
    return joblib.load(path)


if __name__ == "__main__":
    # Test the feature creation.
    import pandas as pd

    train_df = pd.read_csv("data/train.csv")
    vectorizer, X_train_features = fit_features(train_df["Message"])
    save_vectorizer(vectorizer)
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("Train feature matrix shape:", X_train_features.shape)
