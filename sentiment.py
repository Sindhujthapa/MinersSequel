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

# 6. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_dt_prob = dt.predict_proba(X_test)[:, 1]
y_dt_pred = dt.predict(X_test)
auc_dt = roc_auc_score(y_test, y_dt_prob)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_dt_prob)
print(f"AUC (Decision Tree): {auc_dt:.3f}")

# 7. SVC
svc = SVC(probability=True, random_state=42)
svc.fit(X_train, y_train)
y_svc_prob = svc.predict_proba(X_test)[:, 1]
y_svc_pred = svc.predict(X_test)
auc_svc = roc_auc_score(y_test, y_svc_prob)
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_svc_prob)
print(f"AUC (SVC): {auc_svc:.3f}")

# 8. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_lr_prob = lr.predict_proba(X_test)[:, 1]
y_lr_pred = lr.predict(X_test)
auc_lr = roc_auc_score(y_test, y_lr_prob)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_lr_prob)
print(f"AUC (Logistic Regression): {auc_lr:.3f}")

# 9. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_rf_prob = rf.predict_proba(X_test)[:, 1]
y_rf_pred = rf.predict(X_test)
auc_rf = roc_auc_score(y_test, y_rf_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_rf_prob)
print(f"AUC (Random Forest): {auc_rf:.3f}")

# 10. Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_nb_prob = nb.predict_proba(X_test)[:, 1]
y_nb_pred = nb.predict(X_test)
auc_nb = roc_auc_score(y_test, y_nb_prob)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_nb_prob)
print(f"AUC (Naive Bayes): {auc_nb:.3f}")

# 11. Combined ROC Plot
plt.figure(figsize=(10, 7))
plt.plot(fpr_vader, tpr_vader, label=f'VADER (AUC = {auc_vader:.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot(fpr_svc, tpr_svc, label=f'SVC (AUC = {auc_svc:.2f})')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {auc_nb:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Learning Curves
models = {
    'Decision Tree': dt,
    'SVC': svc,
    'Logistic Regression': lr,
    'Random Forest': rf,
    'Naive Bayes': nb
}

plt.figure(figsize=(10, 7))
for name, model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='roc_auc', n_jobs=-1,
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
