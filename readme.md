## 0. Download data

* Check which [embeddings models](https://sbert.net/docs/sentence_transformer/pretrained_models.html) are multilingual, and which languages are best suited for those models
* [Steam API](https://partner.steamgames.com/doc/store/getreviews) for sentiment analysis (multiple languages, according to prior bullet)
  * List of games with total number of reviews [here](https://steamdb.info/stats/gameratings/?sort=reviews_desc)
* Look for sentiment datasets for additional testing (multiple languages, according to the first bullet)

List of games to scrape. Please get 500 reviews per game. For now, only English. **Pending multilingual embedding for other languages.**

We are missing the scrape code. Whoever creates the code, please make sure to use this type of URL format:
https://store.steampowered.com/appreviews/1128000?json=1&filter=all&language=english&day_range=365

We need something to identify each individual review (probably "recommendationid"), and then get the language, review, and sentiment ("voted_up"). 
**NOTE: a parameter called "cursor" must be passed in the URL to get additional reviews.** Check API documentation for details.

After that, check if you have an even split if your game has a mixed rating, mostly 1s if positive, and mostly 0s if negative.

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
