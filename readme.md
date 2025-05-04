# Miners Sequel
# Sentiment Analysis using Steam Reviews
## Team Members: Nilay, Mariia, Vasco, Sindhuj
## Problem Statement
Steam hosts millions of user reviews in various languages, shaping both game visibility and perceived quality. However, the unstructured nature of this data presents challenges for systematically evaluating user sentiment.

Our project aims to:

1. Build and evaluate sentiment classification models to determine whether a review expresses positive or negative sentiment.

2. Compare traditional rule-based approaches (e.g., VADER) with modern embedding-based models (e.g., Sentence Transformers).

3. Assess multilingual model performance, focusing on scalability to languages beyond English, starting with Spanish.

4. Identify the most accurate and scalable approach for sentiment analysis across languages and platforms.

5. Optimize model parameters and dataset choices for each technique.

6. Test model robustness by introducing non-gaming data to evaluate domain sensitivity of embedding-based models.

## Data Sources 
We used the [Steam API](https://partner.steamgames.com/doc/store/getreviews) to scrape game reviews directly from the gaming website. To manage the balance of upvotes and ensure that we have a sizeable amount of reviews, we choose our games specifically. Specifically, these games had, at least, more than 10,000 recent reviews. The following table shows the games considered.

| Game Name (App Id)                   | Review Score | Review Rating        | Category             | 
|------------------------------|--------------|----------------------|----------------------| 
| Schedule I (3164500)                     | 97%          | Extremely Positive                | Crime               |
| Balatro (2379780)      | 96%          | Extremely Positive             | Cards/Poker           | 
| WEBFISHING (3146520)      | 96%          | Extremely Positive             | Fishing, Multi Player | 
| NBA 2K20 (1089350)                     | 50%          | Mixed                | Sports               |
| Kerbal Space Program 2 (954850)      | 30%          | Negative             | Simulation           | 
| Monster Hunter Wilds (2246340)      | 59%          | Mixed                | Action, Multi Player | 
| Cube World (1128000)                | 37%          | Mostly Negative   | Open World   | 
| Terraria (105600)                    | 97%          | Extremely Positive   | Sandbox, Survival    | 
| Portal 2 (620)                     | 98%          | Extremely Positive   | Puzzle, Adventure    | 
| The Crew 2 (646910)                    | 74%          | Positive             | Driving            |
| Star Wars: Battlefront Classic Collection (2446550)| 23% | Extremely Negative| Space FPS            |
| Mafia III: Definitive Edition (360430)| 57%          | Mixed                | Action, Open World   | 
| Resident Evil Resistance (952070)                | 38%          | Negative             | Horror, Multiplayer    | 
| Wolcen: Lords of Mayhem (424370)      | 55%          | Mixed                | Action, RPG          | 
| Cities: Skylines II (949230)         | 52%          | Mixed                | City Builder         | 
| Tekken 8 (1778820)                    | 54%          | Mixed                | Fighting             | 
| EA SPORTS™ FIFA 23 (1811260)          | 57%          | Mixed                | Sports               | 
| Call of Duty® (1938090)      | 59%          | Mixed                | FPS, Multiplayer     | 
| Fall Guys (1097150)      | 81%          | Positive                | Battle Royale, Multiplayer     | 

Our combined dataset consisted of the following columns/features:
* **Recommendation ID**: a unique identifier for each review.
* **Language**: the language in which the review was written.
* **Voted up**: indicator for positive reviews.
* **App ID**: the unique identifier for each game.
* **Review**: the raw text of each review.

## Data Cleaning and Preprocessing
Initially, when we attempted to scrape reviews, we noticed that the API had some issues when calling for a new set of reviews. Multiple times, the API returned duplicate reviews. To address this problem, we exploited the fact that the API call could filter across different parameters, like only positive or negative reviews, review recency, by ownership of the game, etc. Iterating over all the possible parameter combinations, we were able to collect at least 500 reviews for each game and compile a little more than 9000 unique reviews in English. We followed the same procedure for Spanish reviews and compiled more than 7000 unique reviews. 

The final English dataset was perfectly balanced. However, the Spanish dataset was initially unbalanced, so we had to create a balanced file by randomly sampling an equal number of positive and negative reviews.

## Limitations
* Some reviews do not consider correct sentence structure. For instance, a review could just contain emojis or single-word responses like trash, legendary, perfect, etc. So certain classifiers that consider context and the placement of words within a sentence could notice inaccuracies or be unusable altogether.
* Certain terminology may be unique to specific games. As a result, several words may be associated with the value of that specific game rather than general sentiments.
* Reviews with nuance in them may not be completely representative of the "upvote" indicator. For instance, when someone uses "I liked this game but...", it could contain information valuable to both positive and negative predictions.
* Reviews can be used to increase a user's account level on Steam. Because of such instances, certain reviews are not genuine/are rushed.



## 1. Basic Sentiment Classifier

* Logistic Regression
* Other Classifier models seen in class
* Based on Polarity scoring (check python libraries like [VADER](https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/04-Sentiment-Analysis.html))

### 1.1 Benchmarks

* AUC, the other curve seen in class
* Learning curve (rate of learning depending on how many data we feed the model)

## 2. Embedding 

* Select some embedding models (english and multilingual)
* Train models with data (check last lecture code for this type of implementation)

### 2.1 Observations relevant for Krueger

* Does embedding capture sentiment? 
* If it does, can we compare between them (the embedding models) using the usual metrics?
* Can we train a embedding model with more than 1 language (spanish, french - check hierarchy of languages)

### 2.2 Benchmarks

* Same as 1.1
* Check what happens if: 
  1. Use English embedding with English data, 
  2. multilingual embedding with Spanish (other language) data, 
  3. multilingual with English

## Comparisons between 1 and 2

* Check how they compare with the Steam data
* Check how they do with data from other settings and how they compare
