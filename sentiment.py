import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# 1. Load dataset
df = pd.read_csv('/Users/masha/Desktop/Data mining (texas)/MinersSequel/steam_reviews.csv')
print("Data loaded.")
print(df.head())

# 2. VADER Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
df['vader_score'] = df['review'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
df['vader_label'] = df['vader_score'].apply(lambda x: 1 if x >= 0 else 0)

# 3. VADER ROC + AUC
y_true = df['voted_up']
y_scores = df['vader_score']
auc_vader = roc_auc_score(y_true, y_scores)
print(f"AUC (VADER): {auc_vader:.3f}")
fpr_vader, tpr_vader, _ = roc_curve(y_true, y_scores)

# 4. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['review'].astype(str))
y = df['voted_up']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_dt_prob = dt.predict_proba(X_test)[:, 1]
y_dt_pred = dt.predict(X_test)
auc_dt = roc_auc_score(y_test, y_dt_prob)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_dt_prob)
print(f"AUC (Decision Tree): {auc_dt:.3f}")

# 7. SVC Classifier
svc = SVC(probability=True, random_state=42)
svc.fit(X_train, y_train)
y_svc_prob = svc.predict_proba(X_test)[:, 1]
y_svc_pred = svc.predict(X_test)
auc_svc = roc_auc_score(y_test, y_svc_prob)
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_svc_prob)
print(f"AUC (SVC): {auc_svc:.3f}")

# 8. Combined ROC Plot
plt.figure(figsize=(7, 6))
plt.plot(fpr_vader, tpr_vader, label=f'VADER (AUC = {auc_vader:.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot(fpr_svc, tpr_svc, label=f'SVC (AUC = {auc_svc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curve: VADER vs. Decision Tree vs. SVC')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Confusion Matrices

# VADER
cm_vader = confusion_matrix(y_true, df['vader_label'])
plt.figure(figsize=(5, 4))
sns.heatmap(cm_vader, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - VADER')
plt.tight_layout()
plt.show()

print("Classification Report - VADER")
print(classification_report(y_true, df['vader_label'], target_names=['Negative', 'Positive']))

# Decision Tree
cm_dt = confusion_matrix(y_test, y_dt_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree')
plt.tight_layout()
plt.show()
print("Classification Report - Decision Tree")
print(classification_report(y_test, y_dt_pred, target_names=['Negative', 'Positive']))

# SVC
cm_svc = confusion_matrix(y_test, y_svc_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVC')
plt.tight_layout()
plt.show()
print("Classification Report - SVC")
print(classification_report(y_test, y_svc_pred, target_names=['Negative', 'Positive']))

# 10. Learning Curves
# Decision Tree learning curve
train_sizes_dt, train_scores_dt, test_scores_dt = learning_curve(
    dt, X, y, cv=5, scoring='roc_auc', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# SVC learning curve
train_sizes_svc, train_scores_svc, test_scores_svc = learning_curve(
    svc, X, y, cv=5, scoring='roc_auc', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Combined plot
plt.figure(figsize=(7, 6))
plt.plot(train_sizes_dt, train_scores_dt.mean(axis=1), label="Training AUC - DT")
plt.plot(train_sizes_dt, test_scores_dt.mean(axis=1), label="Validation AUC - DT")
plt.plot(train_sizes_svc, train_scores_svc.mean(axis=1), label="Training AUC - SVC")
plt.plot(train_sizes_svc, test_scores_svc.mean(axis=1), label="Validation AUC - SVC")

plt.xlabel("Training Examples")
plt.ylabel("AUC")
plt.title("Learning Curves - Decision Tree vs. SVC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
