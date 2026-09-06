import json
import os
import joblib
import streamlit as st

from src.preprocess import clean_text


# =========================================================
# PATHS
# =========================================================

MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
RESULTS_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Optimized Spam Detection System",
    page_icon="📩",
    layout="centered"
)


# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

@st.cache_resource
def load_artifacts():
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    best_model_name = results["best_model"]

    tuned_model_path = os.path.join(
        MODELS_DIR,
        f"{best_model_name}_tuned.pkl"
    )

    base_model_path = os.path.join(
        MODELS_DIR,
        f"{best_model_name}.pkl"
    )

    if os.path.exists(tuned_model_path):
        model_path = tuned_model_path
    else:
        model_path = base_model_path

    model = joblib.load(model_path)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer, results


# =========================================================
# TITLE
# =========================================================

st.title("📩 Optimized Spam Detection System")

st.write(
    "Enter a message below and the trained machine learning model "
    "will predict whether it is Spam or Ham."
)


# =========================================================
# LOAD ARTIFACTS
# =========================================================

try:
    model, vectorizer, evaluation_results = load_artifacts()

except Exception as e:
    st.error("Could not load the trained model files.")
    st.write(e)
    st.stop()


# =========================================================
# MESSAGE INPUT
# =========================================================

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a free prize..."
)


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔍 Check Message"):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        cleaned_message = clean_text(message)

        features = vectorizer.transform([cleaned_message])

        prediction = model.predict(features)[0]

        st.subheader("Prediction")

        if prediction == "spam":
            st.error("🚨 SPAM")

        else:
            st.success("✅ HAM (Not Spam)")

        # -------------------------------------------------
        # Probabilities
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(features)[0]

            classes = model.classes_

            probability_dict = dict(zip(classes, probabilities))

            spam_probability = probability_dict.get("spam", 0)
            ham_probability = probability_dict.get("ham", 0)

            st.write(
                f"**Spam Probability:** {spam_probability:.2%}"
            )

            st.progress(float(spam_probability))

            st.write(
                f"**Ham Probability:** {ham_probability:.2%}"
            )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("🤖 Model Information")

best_model = evaluation_results["best_model"]

st.write(
    f"**Selected Best Model:** `{best_model}`"
)

st.write(
    "**Selection Method:** "
    "5-fold cross-validated Spam Precision on the training set."
)


# =========================================================
# EVALUATION RESULTS
# =========================================================

st.subheader("📊 Model Performance")

best_results = evaluation_results["results"][best_model]
metrics = best_results["test_metrics"]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}"
    )

    st.metric(
        "Precision",
        f"{metrics['precision']:.2%}"
    )

with col2:
    st.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

    st.metric(
        "F1 Score",
        f"{metrics['f1']:.2%}"
    )


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("🔢 Confusion Matrix")

cm = metrics["confusion_matrix"]

confusion_data = {
    "Actual / Predicted": [
        "Ham → Ham",
        "Ham → Spam",
        "Spam → Ham",
        "Spam → Spam"
    ],
    "Count": [
        cm["tn_ham_as_ham"],
        cm["fp_ham_as_spam"],
        cm["fn_spam_as_ham"],
        cm["tp_spam_as_spam"]
    ]
}

st.table(confusion_data)

