import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# --- Function to plot learning curve ---
def plot_learning_curve_growing_train(X, y, model_name, steps=10, min_test_size=200):
    sizes = []
    aucs = []
    aps = []
    accs = []

    total_samples = len(X)
    max_train_size = total_samples - min_test_size
    step_sizes = np.linspace(100, max_train_size, steps, dtype=int)

    print(f"\n--- Learning Curve for {model_name} ---")
    for size in step_sizes:
        print(f"Training with {size} samples...")

        # Split: first `size` for training, rest for testing
        X_train = X[:size]
        y_train = y[:size]
        X_test = X[size:]
        y_test = y[size:]

        # Skip if only one class in train or test (can't compute AUC)
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
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
    plt.title(f"Learning Curve: {model_name}")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and show plot
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    plt.savefig(f"{safe_name}_learning_curve.png")
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

    plot_learning_curve_growing_train(X_emb, y, model_name=label)
