import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, ConfusionMatrixDisplay, average_precision_score
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# --- Updated Evaluation Function with ROC data collection ---
def evaluate_model_with_roc(X, y, model_name, roc_data):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=5,
        scoring='roc_auc',
        verbose=0,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[model_name] = (fpr, tpr, auc)

    print(f"\n--- {model_name} ---")
    print(f"AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print("Best parameters:", grid_search.best_params_)

    return auc, ap

# --- Load datasets ---
df_en = pd.read_csv("steam_reviews_unique.csv", encoding="utf-8-sig")
X_en = df_en["review"].astype(str).tolist()
y_en = df_en["voted_up"].astype(int)

df_es = pd.read_csv("steam_reviews_unique_esp.csv", encoding="utf-8-sig")
X_es = df_es["review"].astype(str).tolist()
y_es = df_es["voted_up"].astype(int)

# --- Initialize models ---
models = {
    "MiniLM-English": SentenceTransformer("all-MiniLM-L6-v2"),
    "MiniLM-Multilingual (EN)": SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
    "MiniLM-Multilingual (ES)": SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
}

# --- Main Execution ---
roc_data = {}

print("Encoding English reviews with English model...")
emb_en = models["MiniLM-English"].encode(X_en, show_progress_bar=True)
evaluate_model_with_roc(emb_en, y_en, "MiniLM-English", roc_data)

print("\nEncoding English reviews with Multilingual model...")
emb_en_multi = models["MiniLM-Multilingual (EN)"].encode(X_en, show_progress_bar=True)
evaluate_model_with_roc(emb_en_multi, y_en, "MiniLM-Multilingual (EN)", roc_data)

print("\nEncoding Spanish reviews with Multilingual model...")
emb_es = models["MiniLM-Multilingual (ES)"].encode(X_es, show_progress_bar=True)
evaluate_model_with_roc(emb_es, y_es, "MiniLM-Multilingual (ES)", roc_data)

# --- Combined ROC Curve Plot ---
plt.figure(figsize=(10, 7))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.title("ROC Curves for SentenceTransformer Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


