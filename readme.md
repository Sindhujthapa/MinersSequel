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
* AUC
* Average Precision
* ROC curve
* Learning curve (rate of learning depending on how much data we feed the model)

## 2. Embedding 

### 2.1 Benchmarks
* AUC
* Average Precision
* ROC curve
* Learning curve (rate of learning depending on how much data we feed the model)

### 2.2 Methodology
To expand our analysis from the earlier models, we tried to use the data to create embeddings based on Sentence Bidirectional Encoder Representations from Transformers (SBERT). We chose these embedding models as they consider each word within the context of its position in a sentence. We use the following versions of the model:
1. all-MiniLM-L6-v2: This is a relatively fast model that can be used as a baseline for embedding text in the English language.
2. paraphrase-Multilingual-MiniLM-L12-v2: This is a longer, multilingual model used in this context for English and Spanish.

After embedding, we made predictions using Logistic Regression, as this was the best/most consistent classifier for our dataset. To optimize the classifier, we fine-tuned the parameters C (inverse of regularization strength), the penalty size, and the solver type. We finally consider the area under the curve (AUC) as 'scoring' when finding our best model using GridsearchCV.

### 2.3 Main Findings
After running both types of models, we noticed the following performance across the benchmarks:

| Metric             | MiniLM-English                                | MiniLM-Multilingual (EN)                    | MiniLM-Multilingual (ES)                    |
|--------------------|-----------------------------------------------|---------------------------------------------|---------------------------------------------|
| AUC                | 0.8861                                        | 0.8890                                      | 0.8712                                      |
| Average Precision  | 0.8999                                        | 0.8997                                      | 0.8721                                      |
| Best Parameters    | {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'} | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'} | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'} |

The next plot shows the ROC curve:

![image](ROC_curve.jpeg)

### 2.4 Robustness checks
#### i. Performance across other data (movie reviews)
We downloaded a dataset from Hugging Face that lists movie reviews with sentiments from IMDb. We trained the model using the English data from the Steam reviews and tested it on the IMDb dataset. We noticed the following performance across the benchmarks:
| Evaluation Setting                      | AUC    | Average Precision | Best Logistic Regression Params                     |
|----------------------------------------|--------|-------------------|-----------------------------------------------------|
| Train: Steam<br>Test: IMDB             | 0.8168 | 0.8200            | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}  |
| Train: EN (Steam)<br>Test: ES (Steam)  | 0.8726 | 0.8701            | {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}      |
| Train: ES (Steam)<br>Test: EN (Steam)  | 0.8743 | 0.8702            | {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'}    |


#### ii. Performance across training set size.
To test how our model's performance changes with training group size, we plot these benchmarks for a fixed test size across increasing training sizes. All samples are randomly selected and stratified to maintain balance in observations. 
Here's how our benchmarks changed with increasing training set size:
![image](learning_curve_train_size_en.png)

![image](Learning_curve_train_size_es.png)

#### iii. Performance across sample sizes.
To test the minimum amount of equally split sample data we need to optimize our model's performance, we plot the benchmarks across increasing sample sizes.
![image](Learning_curve_sample_size_english.png)

![image](Learning_curve_sample_size_spanish.png)

### 2.2 Benchmarks

* Same as 1.1
* Check what happens if: 
  1. Use English embedding with English data, 
  2. multilingual embedding with Spanish (other language) data, 
  3. multilingual with English

## Comparisons between 1 and 2

* Check how they compare with the Steam data
* Check how they do with data from other settings and how they compare
