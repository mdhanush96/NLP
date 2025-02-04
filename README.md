# NLP
# Twitter Sentiment Analysis

## Overview
This Python script performs sentiment analysis on Twitter data using Natural Language Processing (NLP) techniques and machine learning. It utilizes TF-IDF vectorization and Logistic Regression to classify tweets as Negative, Neutral, or Positive.

## Features
- Data loading from an external dataset
- Text preprocessing (lowercasing, stopwords removal, lemmatization, and tokenization)
- TF-IDF vectorization
- Logistic Regression model for sentiment classification
- Model evaluation using accuracy, classification report, and confusion matrix visualization

## Requirements
Ensure you have Python installed along with the following libraries:

```bash
pip install pandas numpy nltk scikit-learn seaborn matplotlib
```

Additionally, download necessary NLTK datasets:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## How to Use
1. Run the script:
```bash
python sentiment_analysis.py
```
2. The script will:
   - Load the dataset
   - Preprocess text data
   - Vectorize text using TF-IDF
   - Train a Logistic Regression model
   - Evaluate the model and display results

## Expected Output
- Accuracy score of the model
- Classification report detailing precision, recall, and F1-score
- Confusion matrix visualization

## Dataset
The script uses a dataset from:
[Twitter Sentiment Analysis Dataset](https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv)

Ensure the dataset contains `tweet` and `label` columns.

## License
This project is open-source and available under the MIT License.

