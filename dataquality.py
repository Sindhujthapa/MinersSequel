import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['review', 'voted_up']].dropna()
    df.columns = ['text', 'label']
    return df['text'], df['label']

def create_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])

def define_param_grid():
    return {
        'tfidf__min_df': [1, 2, 5],
        'tfidf__max_df': [0.5, 0.75, 1.0],
        'tfidf__max_features': [1000, 3000],
        'clf__C': [0.01, 0.1, 1, 10]
    }

def perform_grid_search(pipeline, param_grid, X_train, y_train):
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    return grid

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, y_prob):.4f}")

def print_best_params(params):
    print("\n--- Best Parameters Found ---")
    print(f"min_df: {params['tfidf__min_df']}")
    print(f"max_df: {params['tfidf__max_df']}")
    print(f"max_features: {params.get('tfidf__max_features', 'Not specified')}")
    print(f"C (inverse regularization strength): {params['clf__C']}")

def main():
    X, y = load_data('steam_reviews_unique.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_pipeline()
    param_grid = define_param_grid()
    grid = perform_grid_search(pipeline, param_grid, X_train, y_train)

    best_model = grid.best_estimator_
    print_best_params(grid.best_params_)

    evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()