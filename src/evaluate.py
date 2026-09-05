import os
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

MODEL_FILES = {
    "naive_bayes": "naive_bayes.pkl",
    "logistic_regression": "logistic_regression.pkl",
    "svm": "svm.pkl",
}

PARAM_GRIDS = {
    "naive_bayes": {
        "alpha": [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "fit_prior": [True, False],
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
        "class_weight": [None, "balanced"],
    },
    "svm": {
        "estimator__C": [0.01, 0.1, 0.5, 1, 5, 10, 50],
        "estimator__class_weight": [None, "balanced"],
    },
}

SPAM_PRECISION_SCORER = make_scorer(precision_score, pos_label="spam", zero_division=0)


def load_saved_artifacts():
    vectorizer = joblib.load(VECTORIZER_PATH)
    return vectorizer


def load_split():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return (train_df["Message"], train_df["Label"],
            test_df["Message"], test_df["Label"])


def fresh_estimator(name):
    if name == "naive_bayes":
        return MultinomialNB()
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if name == "svm":
        return CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3)
    raise ValueError(name)


def tune_model(name, X_train_features, y_train):
    grid = GridSearchCV(
        fresh_estimator(name),
        PARAM_GRIDS[name],
        cv=5,
        scoring=SPAM_PRECISION_SCORER,
        n_jobs=-1,
    )
    grid.fit(X_train_features, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def score_on_test(model, X_test_features, y_test):
    preds = model.predict(X_test_features)
    cm = confusion_matrix(y_test, preds, labels=["ham", "spam"])
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, pos_label="spam"),
        "recall": recall_score(y_test, preds, pos_label="spam"),
        "f1": f1_score(y_test, preds, pos_label="spam"),
        "confusion_matrix": {
            "tn_ham_as_ham": int(cm[0][0]),
            "fp_ham_as_spam": int(cm[0][1]),
            "fn_spam_as_ham": int(cm[1][0]),
            "tp_spam_as_spam": int(cm[1][1]),
        },
    }


def main():
    vectorizer = load_saved_artifacts()
    X_train_text, y_train, X_test_text, y_test = load_split()

    X_train_features = vectorizer.transform(X_train_text)
    X_test_features = vectorizer.transform(X_test_text)

    tuned = {}
    for name in MODEL_FILES:
        print(f"Tuning {name} (train set only, scored on spam precision)...")
        best_estimator, best_params, cv_best_score = tune_model(
            name, X_train_features, y_train
        )
        tuned[name] = {
            "estimator": best_estimator,
            "best_params": best_params,
            "cv_precision_spam": cv_best_score,
        }

    best_name = max(tuned, key=lambda n: tuned[n]["cv_precision_spam"])
    print(f"\nSelected '{best_name}' based on cross-validated training scores "
          f"(test set not used for this decision).")

    results = {}
    for name, info in tuned.items():
        test_metrics = score_on_test(info["estimator"], X_test_features, y_test)
        results[name] = {
            "test_metrics": test_metrics,
            "tuning": {
                "best_params": info["best_params"],
                "cv_precision_spam": info["cv_precision_spam"],
            },
        }

    table_rows = []
    for name, r in results.items():
        m = r["test_metrics"]
        cm = m["confusion_matrix"]
        table_rows.append({
            "model": name,
            "accuracy": round(m["accuracy"], 4),
            "precision(spam)": round(m["precision"], 4),
            "recall(spam)": round(m["recall"], 4),
            "f1(spam)": round(m["f1"], 4),
            "false_positives": cm["fp_ham_as_spam"],
            "false_negatives": cm["fn_spam_as_ham"],
            "cv_precision(train, tuned)": round(r["tuning"]["cv_precision_spam"], 4),
            "best_params": r["tuning"]["best_params"],
        })
    comparison_df = pd.DataFrame(table_rows).sort_values(
        "cv_precision(train, tuned)", ascending=False
    )

    print("\n" + "=" * 90)
    print("MODEL COMPARISON  (positive class = spam)")
    print("=" * 90)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("CONFUSION MATRICES (test set, tuned/refit models)")
    print("=" * 90)
    for name, r in results.items():
        cm = r["test_metrics"]["confusion_matrix"]
        tn = cm["tn_ham_as_ham"]
        fp = cm["fp_ham_as_spam"]
        fn = cm["fn_spam_as_ham"]
        tp = cm["tp_spam_as_spam"]
        print(f"\n{name}:")
        print(f"  Real ham  -> predicted ham : {tn}")
        print(f"  Real ham  -> predicted spam: {fp}  (false positive: blocked a real message)")
        print(f"  Real spam -> predicted ham : {fn}  (false negative: spam got through)")
        print(f"  Real spam -> predicted spam: {tp}")

    best_test_metrics = results[best_name]["test_metrics"]
    reasoning = (
        f"'{best_name}' was selected using 5-fold cross-validated spam-precision "
        f"on the TRAINING set only ({tuned[best_name]['cv_precision_spam']:.3f}), "
        f"before the test set was touched. Its final, one-time test evaluation "
        f"confirms precision={best_test_metrics['precision']:.3f} and "
        f"f1={best_test_metrics['f1']:.3f}, which is reported here as an unbiased "
        f"estimate of real-world performance -- not as the basis for the selection."
    )
    print("\n" + "=" * 90)
    print(f"BEST MODEL: {best_name}")
    print(reasoning)
    print("=" * 90)

    os.makedirs(MODELS_DIR, exist_ok=True)
    comparison_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "w") as f:
        json.dump({
            "positive_class": "spam",
            "selection_method": "cross-validated spam precision on training set only",
            "results": results,
            "best_model": best_name,
            "reasoning": reasoning,
        }, f, indent=2)

    joblib.dump(tuned[best_name]["estimator"], os.path.join(MODELS_DIR, f"{best_name}_tuned.pkl"))
    print(f"\nSaved {MODELS_DIR}/model_comparison.csv, {MODELS_DIR}/evaluation_results.json, "
          f"and {MODELS_DIR}/{best_name}_tuned.pkl")

    preds = tuned[best_name]["estimator"].predict(X_test_features)
    print(f"\nFull classification report for {best_name} (tuned, final test evaluation):\n")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
