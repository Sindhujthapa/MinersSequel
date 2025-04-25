## 0. Download data

* Check which embeddings models are multilingual, and which languages are best suited for those models
* Steam API for sentiment analysis (multiple languages, according to prior bullet)
* Look for sentiment datasets for additional testing (multiple languages, according to first bullet)

## 1. Basic Sentiment Classifier

* Logistic Regression
* Other Classifier models seen in class
* Based on Polarity scoring (check python libraries like VADER)

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
