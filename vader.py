
import pandas as pd


df = pd.read_csv('/Users/masha/Desktop/Data mining (texas)/MinersSequel/steam_reviews.csv')
print("Data loaded.")
print(df.shape)
print(df.head())


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set up sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Create polarity score column (assumes your review column is called 'review')
df['vader_score'] = df['review'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Create binary sentiment label: 1 = positive, 0 = negative
df['vader_label'] = df['vader_score'].apply(lambda x: 1 if x >= 0 else 0)

# Preview
print(df[['review', 'vader_score', 'vader_label']].head())

