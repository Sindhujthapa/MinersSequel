import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, ConfusionMatrixDisplay, average_precision_score
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.utils import resample
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")


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
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return auc, ap


df_en = pd.read_csv("Scraper and Data/steam_reviews_unique.csv", encoding="utf-8-sig")
X_en = df_en["review"].astype(str).tolist()
y_en = df_en["voted_up"].astype(int)

df_es = pd.read_csv("Scraper and Data/steam_reviews_balanced_esp.csv", encoding="utf-8-sig")
X_es = df_es["review"].astype(str).tolist()
y_es = df_es["voted_up"].astype(int)


models = {
    "MiniLM-English": SentenceTransformer("all-MiniLM-L6-v2"),
    "MiniLM-Multilingual (EN)": SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
    "MiniLM-Multilingual (ES)": SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
}


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

# --- Non-Gaming Reviews Execution ---
def balanced_subsample(df, target_size):
    half = target_size // 2
    df_pos = df[df["voted_up"] == 1]
    df_neg = df[df["voted_up"] == 0]

    df_pos_sampled = resample(df_pos, replace=False, n_samples=half, random_state=42)
    df_neg_sampled = resample(df_neg, replace=False, n_samples=half, random_state=42)

    return pd.concat([df_pos_sampled, df_neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)


dataset = load_dataset("imdb")
imdb = dataset["test"].to_pandas()
imdb = imdb.rename(columns={'label': 'voted_up', 'text': 'review'})

n_en, n_imdb = len(df_en), len(imdb)
target_size = min(n_en, n_imdb)

if n_en > target_size:
    df_en = balanced_subsample(df_en, target_size)
else:
    imdb = balanced_subsample(imdb, target_size)

print(df_en["voted_up"].value_counts())
print(imdb["voted_up"].value_counts())

X_en = df_en["review"].tolist()
y_en = df_en["voted_up"]
X_imdb = imdb["review"].tolist()
y_imdb = imdb["voted_up"]

print("Encoding Steam and IMDB reviews...")
model = SentenceTransformer("all-MiniLM-L6-v2")
X_en_emb = model.encode(X_en, show_progress_bar=True)
X_imdb_emb = model.encode(X_imdb, show_progress_bar=True)

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
    n_jobs=-1,
    verbose=1
)

#Evaluate on IMDB embeddings
print("Training Logistic Regression on Steam embeddings...")
grid_search.fit(X_en_emb, y_en)
best_model = grid_search.best_estimator_

y_proba = best_model.predict_proba(X_imdb_emb)[:, 1]
y_pred = best_model.predict(X_imdb_emb)

auc = roc_auc_score(y_imdb, y_proba)
ap = average_precision_score(y_imdb, y_proba)

print("\n--- Evaluation (Train: Steam | Test: IMDB) ---")
print(f"AUC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print("Best Logistic Regression Params:", grid_search.best_params_)
print("Classification Report:")
print(classification_report(y_imdb, y_pred))

# --- Bias Language Execution ---
df_en = pd.read_csv("Scraper and Data/steam_reviews_unique.csv", encoding="utf-8-sig")
n_en, n_es = len(df_en), len(df_es)
target_size = min(n_en, n_es)

if n_en > target_size:
    df_en = balanced_subsample(df_en, target_size)
else:
    df_es = balanced_subsample(df_es, target_size)

X_en = df_en["review"].tolist()
y_en = df_en["voted_up"]
X_es = df_es["review"].tolist()
y_es = df_es["voted_up"]

print("Encoding reviews with multilingual model (bias estimation)...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
X_en_emb = model.encode(X_en, show_progress_bar=True)
X_es_emb = model.encode(X_es, show_progress_bar=True)

#Evaluate on Spanish embeddings
print("Training Logistic Regression on English embeddings...")
grid_search.fit(X_en_emb, y_en)
best_model = grid_search.best_estimator_

y_proba = best_model.predict_proba(X_es_emb)[:, 1]
y_pred = best_model.predict(X_es_emb)

auc = roc_auc_score(y_es, y_proba)
ap = average_precision_score(y_es, y_proba)

print("\n--- Cross-lingual Evaluation (Train: EN | Test: ES) ---")
print(f"AUC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print("Best Logistic Regression Params:", grid_search.best_params_)
print("Classification Report:")
print(classification_report(y_es, y_pred))


#Evaluate on English embeddings
print("Training Logistic Regression on Spanish embeddings...")
grid_search.fit(X_es_emb, y_es)
best_model = grid_search.best_estimator_

y_proba = best_model.predict_proba(X_en_emb)[:, 1]
y_pred = best_model.predict(X_en_emb)

auc = roc_auc_score(y_en, y_proba)
ap = average_precision_score(y_en, y_proba)

print("\n--- Cross-lingual Evaluation (Train: ES | Test: EN) ---")
print(f"AUC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print("Best Logistic Regression Params:", grid_search.best_params_)
print("Classification Report:")
print(classification_report(y_en, y_pred))

