import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv('/Users/masha/Desktop/Data mining (texas)/MinersSequel/steam_reviews.csv')
print("Data loaded.")
print(df.shape)
print(df.head())

# 2. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['review'].astype(str))
y = df['voted_up']  # True sentiment labels

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Support Vector Classifier
svc = SVC(probability=True, random_state=42)
svc.fit(X_train, y_train)

# 5. Predict and evaluate AUC
y_pred_prob_svc = svc.predict_proba(X_test)[:, 1]
y_pred_class_svc = svc.predict(X_test)
auc_svc = roc_auc_score(y_test, y_pred_prob_svc)
print(f"AUC (SVC): {auc_svc:.3f}")

# 6. Plot ROC curve
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_pred_prob_svc)

plt.figure(figsize=(6, 5))
plt.plot(fpr_svc, tpr_svc, label=f'SVC (AUC = {auc_svc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Support Vector Classifier')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class_svc)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVC')
plt.tight_layout()
plt.show()

# Optional: Print classification report
print(classification_report(y_test, y_pred_class_svc, target_names=['Negative', 'Positive']))

# 8. Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    svc, X, y, cv=5, scoring='roc_auc', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

plt.figure(figsize=(6, 5))
plt.plot(train_sizes, train_scores_mean, label="Training AUC")
plt.plot(train_sizes, test_scores_mean, label="Validation AUC")
plt.xlabel("Training Examples")
plt.ylabel("AUC")
plt.title("Learning Curve - SVC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
