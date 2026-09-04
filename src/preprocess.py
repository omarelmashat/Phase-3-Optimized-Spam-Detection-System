import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')
    df = df.rename(columns={'v1': 'Label', 'v2': 'Message'})
    junk_cols = [c for c in df.columns if c.startswith('Unnamed')]
    for col in junk_cols:
        spill = df[col].fillna('')
        df['Message'] = df['Message'] + spill.apply(lambda s: ' ' + s if s else '')
    df = df.drop(columns=junk_cols)
    return df[['Label', 'Message']]

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def run_preprocessing(path='data/spam.csv'):    
    df = load_data(path)
    df = df[df['Label'].isin(['spam', 'ham'])]
    df = df.drop_duplicates(subset=['Message']).reset_index(drop=True)
    df['Message'] = df['Message'].apply(clean_text)
    df = df[df['Message'].str.len() > 0].reset_index(drop=True)

    X = df['Message']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_preprocessing()

    pd.DataFrame({'Message': X_train, 'Label': y_train}).to_csv('data/train.csv', index=False)
    pd.DataFrame({'Message': X_test, 'Label': y_test}).to_csv('data/test.csv', index=False)
    print("Saved data/train.csv and data/test.csv")