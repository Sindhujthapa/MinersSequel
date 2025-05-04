import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("steam_reviews.csv")  # Ensure this file has 'text' and 'sentiment' columns
texts = df["text"].astype(str).tolist()
labels = df["sentiment"].astype(int).tolist()

# Load pre-trained MiniLM model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode text data
print("Encoding text data...")
embeddings = model.encode(texts, show_progress_bar=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']  # 'liblinear' works better with smaller datasets
}

# Set up grid search
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)

# Fit model
print("Running grid search...")
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\nBest parameters found:", grid_search.best_params_)

# Predict and evaluate
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# AUC Score
auc_score = roc_auc_score(y_test, y_proba)
print(f"AUC Score: {auc_score:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Confusion matrix and classification report
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Positive", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
