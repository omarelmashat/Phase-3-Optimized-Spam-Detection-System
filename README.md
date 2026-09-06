# Optimized Spam Detection System

## Project Overview

The Optimized Spam Detection System is a machine learning project designed to classify text messages as either **Spam** or **Ham (legitimate)**.

The project follows a complete end-to-end machine learning pipeline:

**Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → Deployment**

The system also provides a user-friendly **Streamlit web application** that allows users to enter a message and receive a Spam/Ham prediction.

---

## Project Objectives

* Clean and preprocess the message dataset.
* Convert text messages into numerical features.
* Train multiple machine learning classification models.
* Evaluate and compare model performance.
* Select the best-performing model based on evaluation metrics.
* Deploy the trained model through a Streamlit application.
* Provide an easy-to-use interface for real-time message classification.

---

## Project Structure

```text
Phase-3-Optimized-Spam-Detection-System/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── spam_dataset.csv
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── vectorizer.pkl
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── model_comparison.csv
│   ├── evaluation_results.json
│   └── best model files
│
├── notebooks/
│   └── exploration.ipynb
│
└── src/
    ├── preprocess.py
    ├── features.py
    ├── train.py
    └── evaluate.py
```

---

## Machine Learning Pipeline

### 1. Data Preprocessing

The dataset is loaded and inspected for:

* Missing values
* Invalid labels
* Duplicate or malformed records
* Class distribution

Message text is cleaned and normalized before being used for training.

The dataset is split into training and testing sets using a reproducible split.

### 2. Feature Engineering

Text messages are converted into numerical features using text vectorization techniques such as **TF-IDF** or **Count Vectorization**.

The vectorizer is fitted only on the training data and then reused to transform the test data and new messages.

### 3. Model Training

The project trains multiple classification models, including:

* Multinomial Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)

The trained models and fitted vectorizer are saved in the `models/` directory for later reuse.

### 4. Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Hyperparameter tuning is also performed to improve model performance and select the best model.

---

## Deployment

The project includes a **Streamlit** web application.

The application allows the user to:

1. Enter a message.
2. Process the message using the saved vectorizer.
3. Predict whether the message is Spam or Ham.
4. Display prediction probabilities.
5. View model evaluation information such as the confusion matrix.

The application reuses the already-fitted vectorizer and trained model instead of fitting them again on new user input.

---

## Installation

Clone the repository and open the project folder:

```bash
git clone https://github.com/omarelmashat/Phase-3-Optimized-Spam-Detection-System.git

cd Phase-3-Optimized-Spam-Detection-System
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Full Pipeline

To run preprocessing, training, and evaluation:

```bash
python main.py
```

The pipeline will execute:

```text
Data Preprocessing
        ↓
Feature Engineering & Model Training
        ↓
Model Evaluation & Tuning
```

---

## Running the Streamlit Application

After the required models have been generated, run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Example

### Spam Message

```text
Congratulations! You won a free prize. Click here now!
```

Expected classification:

```text
SPAM
```

### Ham Message

```text
Hey, are we still meeting tomorrow?
```

Expected classification:

```text
HAM
```

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib
* Jupyter Notebook
* Git & GitHub

---

## Team

**TechMaster Academy — Phase 03 / Project 03**

**Group 4**

The project was developed collaboratively using separate roles for:

* Data & Preprocessing
* Feature Engineering & Model Training
* Model Evaluation & Tuning
* Integration & Deployment

---

## Conclusion

The Optimized Spam Detection System provides a complete machine learning workflow for detecting spam messages.

The project combines data preprocessing, feature engineering, multiple classification models, evaluation and tuning, and a deployed Streamlit application into one integrated system.
