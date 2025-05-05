import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score
)
from sklearn.utils import resample

def balanced_subsample(df, target_size):
    half = target_size // 2
    df_pos = df[df["voted_up"] == 1]
    df_neg = df[df["voted_up"] == 0]

    df_pos_sampled = resample(df_pos, replace=False, n_samples=half, random_state=42)
    df_neg_sampled = resample(df_neg, replace=False, n_samples=half, random_state=42)

    return pd.concat([df_pos_sampled, df_neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# VADER sentiment scoring
def compute_vader_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['vader_score'] = df['review'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return df

# Choose best threshold based on F1 score (TRAIN ONLY)
def determine_best_threshold(df_train, y_col='voted_up', score_col='vader_score'):
    thresholds = np.linspace(-1, 1, 100)
    y_true = df_train[y_col]
    best_thresh = max(thresholds, key=lambda t: f1_score(y_true, (df_train[score_col] >= t).astype(int)))
    return best_thresh

# TF-IDF vectorization
def vectorize_text(df, text_col='review'):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df[text_col].astype(str))
    return X, vectorizer

# Evaluate VADER scores
def evaluate_vader(df_test, best_thresh):
    y_true = df_test['voted_up']
    y_pred = (df_test['vader_score'] >= best_thresh).astype(int)
    auc = roc_auc_score(y_true, df_test['vader_score'])
    fpr, tpr, _ = roc_curve(y_true, df_test['vader_score'])
    return y_true, y_pred, df_test['vader_score'], fpr, tpr, auc

# Train models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB()
    }

    model_outputs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        model_outputs[name] = (y_test, y_pred, y_prob, fpr, tpr, auc)
        print(f"AUC ({name}): {auc:.3f}")
    return models, model_outputs

# === MAIN PIPELINE ===
if __name__ == '__main__':
    # Load training data from Steam reviews
    df_train = pd.read_csv("steam_reviews_unique.csv")
    df_train = compute_vader_sentiment(df_train)

    # Load IMDB test data
    imdb = load_dataset("imdb")['test'].to_pandas()
    imdb = imdb.rename(columns={'label': 'voted_up', 'text': 'review'})

    # Balance IMDB test data
    imdb_bal = balanced_subsample(imdb, target_size=min(len(df_train), len(imdb)))
    imdb_bal = compute_vader_sentiment(imdb_bal)

    # Determine VADER threshold using Steam data only
    best_thresh = determine_best_threshold(df_train)

    # Vectorize using Steam training data only
    X_train, vectorizer = vectorize_text(df_train)
    y_train = df_train['voted_up']

    X_test = vectorizer.transform(imdb_bal['review'])
    y_test = imdb_bal['voted_up']

    # Evaluate VADER on IMDB
    imdb_bal['vader_label'] = (imdb_bal['vader_score'] >= best_thresh).astype(int)
    y_vader_true, y_vader_pred, vader_scores, fpr_vader, tpr_vader, auc_vader = evaluate_vader(imdb_bal, best_thresh)
    model_outputs = {'VADER': (y_vader_true, y_vader_pred, vader_scores, fpr_vader, tpr_vader, auc_vader)}

    # Train and evaluate models
    models, outputs = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    model_outputs.update(outputs)

    # Output classification reports
    for name, (y_true, y_pred, _, _, _, _) in model_outputs.items():
        print(f"\nClassification Report - {name}")
        print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
