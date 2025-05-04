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
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# 1. Load dataset
df = pd.read_csv('/Users/masha/Desktop/Data mining (texas)/MinersSequel/steam_reviews_unique.csv')
print("Data loaded.")
print(df.head())

# 2. VADER Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
df['vader_score'] = df['review'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
df['vader_label'] = df['vader_score'].apply(lambda x: 1 if x >= 0 else 0)

# 3. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['review'].astype(str))
y = df['voted_up']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. VADER Evaluation on test set
vader_test = df.iloc[y_test.index]
y_vader_test = y_test
vader_pred_test = (vader_test['vader_score'] >= 0).astype(int)
auc_vader = roc_auc_score(y_vader_test, vader_test['vader_score'])
fpr_vader, tpr_vader, _ = roc_curve(y_vader_test, vader_test['vader_score'])
print(f"AUC (VADER): {auc_vader:.3f}")

# 6. Model training and evaluation
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Naive Bayes': MultinomialNB()
}

model_outputs = {
    'VADER': (y_vader_test, vader_pred_test, vader_test['vader_score'], fpr_vader, tpr_vader, auc_vader)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    model_outputs[name] = (y_test, y_pred, y_prob, fpr, tpr, auc)
    print(f"AUC ({name}): {auc:.3f}")

# 7. Confusion Matrices and Classification Reports
for name, (y_true_m, y_pred_m, _, _, _, _) in model_outputs.items():
    cm = confusion_matrix(y_true_m, y_pred_m)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

    print(f"Classification Report - {name}")
    print(classification_report(y_true_m, y_pred_m, target_names=['Negative', 'Positive']))

# 8. Combined ROC Plot
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

# 9. Learning Curves (on training set only)
plt.figure(figsize=(10, 7))
for name, model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    plt.plot(train_sizes, test_scores.mean(axis=1), label=f'Validation AUC - {name}')

plt.xlabel("Training Set Size")
plt.ylabel("AUC Score")
plt.title("Learning Curves for All Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
