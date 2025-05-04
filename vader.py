import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('/Users/masha/Desktop/Data mining (texas)/MinersSequel/steam_reviews.csv')
print("Data loaded.")


print(df.head())

# 2. Apply VADER sentiment analysis
analyzer = SentimentIntensityAnalyzer()
df['vader_score'] = df['review'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
df['vader_label'] = df['vader_score'].apply(lambda x: 1 if x >= 0 else 0)

# 3. Evaluate VADER predictions using ROC/AUC
y_true = df['voted_up']          # Actual user rating
y_scores = df['vader_score']     # VADER compound score

# Compute AUC
auc_vader = roc_auc_score(y_true, y_scores)
print(f"AUC (VADER): {auc_vader:.3f}")

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)

# 4. Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'VADER (AUC = {auc_vader:.2f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - VADER Sentiment')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()