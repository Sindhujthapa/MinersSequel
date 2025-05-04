import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# --- Function to plot learning curve ---
def plot_random_stratified_learning_curve(X, y, model_name, steps=10, fixed_test_size=500):
    sizes = []
    aucs = []
    aps = []
    accs = []

    total_samples = len(X)
    assert total_samples > fixed_test_size + 100, "Not enough data for learning curve."

    # Step 1: stratified random split into test and pool
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=fixed_test_size, stratify=y, random_state=42
    )

    step_sizes = np.linspace(100, len(X_pool) - 10, steps, dtype=int)

    print(f"\n--- Learning Curve for {model_name} (Fixed 500 test set) ---")
    for size in step_sizes:
        print(f"Training with {size} samples...")

        try:
            X_train, _, y_train, _ = train_test_split(
                X_pool, y_pool, train_size=size, stratify=y_pool, random_state=size
            )
        except ValueError as e:
            print(f"Skipping size {size} due to stratification error: {e}")
            continue

        if len(set(y_train)) < 2:
            print(f"Skipping size {size} due to class imbalance.")
            continue

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

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, aucs, label="AUC")
    plt.plot(sizes, aps, label="Average Precision")
    plt.plot(sizes, accs, label="Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.title(f"Learning Curve (Stratified, Disjoint Test): {model_name}")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and show plot
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    plt.savefig(f"{safe_name}_stratified_learning_curve.png")
    plt.show(block=True)

# --- Load Data ---
datasets = [
    ("steam_reviews_unique.csv", "Multilingual MiniLM (English)"),
    ("steam_reviews_balanced_esp.csv", "Multilingual MiniLM (Spanish Balanced)")
]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# --- Run for each dataset ---
for filename, label in datasets:
    print(f"\nLoading dataset: {filename}")
    df = pd.read_csv(filename, encoding="utf-8-sig")
    X_text = df["review"].astype(str).tolist()
    y = df["voted_up"].astype(int)

    print(f"Encoding reviews using {label} model...")
    X_emb = model.encode(X_text, show_progress_bar=True)

    plot_random_stratified_learning_curve(X_emb, y, model_name=label, steps=10, fixed_test_size=500)
