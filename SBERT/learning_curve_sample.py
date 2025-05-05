import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(__file__)) 
filepath1 = os.path.join(base_dir, 'Scraper and Data', 'steam_reviews_unique.csv')

filepath2 = os.path.join(base_dir, 'Scraper and Data', 'steam_reviews_balanced_esp.csv')

def plot_learning_curve(X, y, model_name, steps=10):
    sizes = []
    aucs = []
    aps = []
    accs = []

    total_samples = len(X)
    step_sizes = [int(total_samples * i / steps) for i in range(1, steps + 1)]

    print(f"\n--- Learning Curve for {model_name} ---")
    for size in step_sizes:
        print(f"Training with {size} samples...")

        X_subset = X[:size]
        y_subset = y[:size]

        X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, stratify=y_subset, random_state=42)

        clf = LogisticRegression(max_iter=1000, solver="liblinear")
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        sizes.append(size)
        aucs.append(auc)
        aps.append(ap)
        accs.append(acc)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, aucs, label="AUC")
    plt.plot(sizes, aps, label="Average Precision")
    plt.plot(sizes, accs, label="Accuracy")
    plt.xlabel("Number of samples")
    plt.ylabel("Score")
    plt.title(f"Learning Curve: {model_name}")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

datasets = [
    (filepath1, "Multilingual MiniLM (English)"),
    (filepath2, "Multilingual MiniLM (Spanish - Balanced)")
]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

for filename, label in datasets:
    print(f"\nLoading dataset: {filename}")
    df = pd.read_csv(filename, encoding="utf-8-sig")
    X = df["review"].astype(str).tolist()
    y = df["voted_up"].astype(int)

    print(f"Encoding reviews using {label} model...")
    X_emb = model.encode(X, show_progress_bar=True)

    plot_learning_curve(X_emb, y, label)
    y = df["voted_up"].astype(int)

    print(f"Encoding reviews using {label} model...")
    X_emb = model.encode(X, show_progress_bar=True)

    plot_learning_curve(X_emb, y, label)
