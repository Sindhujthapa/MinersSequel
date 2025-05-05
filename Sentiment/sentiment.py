import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Data preview:")
    print(df.head())
    return df


def compute_vader_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['vader_score'] = df['review'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return df

def determine_best_threshold(df_train, y_col='voted_up', score_col='vader_score'):
    thresholds = np.linspace(-1, 1, 100)
    y_true = df_train[y_col]
    best_thresh = max(thresholds, key=lambda t: f1_score(y_true, (df_train[score_col] >= t).astype(int)))
    return best_thresh

def vectorize_text(df, text_col='review'):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df[text_col].astype(str))
    return X, vectorizer


def split_data(df):
    return train_test_split(df, test_size=0.2, random_state=42, stratify=df['voted_up'])

def evaluate_vader(df_test, best_thresh):
    y_true = df_test['voted_up']
    y_pred = (df_test['vader_score'] >= best_thresh).astype(int)
    auc = roc_auc_score(y_true, df_test['vader_score'])
    fpr, tpr, _ = roc_curve(y_true, df_test['vader_score'])
    return y_true, y_pred, df_test['vader_score'], fpr, tpr, auc

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

def plot_confusion_matrices(model_outputs):
    for name, (y_true, y_pred, _, _, _, _) in model_outputs.items():
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.show()

        print(f"Classification Report - {name}")
        print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

def plot_roc_curves(model_outputs):
    plt.figure(figsize=(10, 7))
    for name, (_, _, _, fpr, tpr, auc) in model_outputs.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_learning_curves(models, X_train, y_train):
    plt.figure(figsize=(10, 7))
    colors = plt.colormaps.get_cmap('tab10')
    for idx, (name, model) in enumerate(models.items()):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)
        color = colors(idx)
        plt.plot(train_sizes, train_mean, label=f'Train AUC - {name}', linestyle='-', color=color)
        plt.plot(train_sizes, test_mean, label=f'Validation AUC - {name}', linestyle='--', color=color)

    plt.xlabel("Training Set Size")
    plt.ylabel("AUC Score")
    plt.title("Learning Curves for All Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = load_data('steam_reviews_unique.csv')

    df = compute_vader_sentiment(df)

    df_train, df_test = split_data(df)

    best_thresh = determine_best_threshold(df_train)

    df_train['vader_label'] = (df_train['vader_score'] >= best_thresh).astype(int)
    df_test['vader_label'] = (df_test['vader_score'] >= best_thresh).astype(int)

    X_train, vectorizer = vectorize_text(df_train)
    X_test = vectorizer.transform(df_test['review'].astype(str))
    y_train = df_train['voted_up']
    y_test = df_test['voted_up']

    y_vader_true, y_vader_pred, vader_scores, fpr_vader, tpr_vader, auc_vader = evaluate_vader(df_test, best_thresh)
    model_outputs = {'VADER': (y_vader_true, y_vader_pred, vader_scores, fpr_vader, tpr_vader, auc_vader)}

    models, outputs = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    model_outputs.update(outputs)

    plot_confusion_matrices(model_outputs)
    plot_roc_curves(model_outputs)
    plot_learning_curves(models, X_train, y_train)
