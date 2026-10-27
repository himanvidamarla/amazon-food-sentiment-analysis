# Project Architecture

## 1. Overview

The Amazon Fine Food Reviews Sentiment Analysis project processes customer reviews using Natural Language Processing (NLP). It applies two sentiment analysis approaches, VADER and RoBERTa, and compares their outputs with customer star ratings.

## 2. System Architecture

```text
Amazon Fine Food Reviews Dataset
              |
              v
       Data Loading (Pandas)
              |
              v
      Data Cleaning and Sampling
              |
              v
  Exploratory Data Analysis (EDA)
              |
              v
       Review Text Processing
              |
       +------+------+
       |             |
       v             v
 VADER Analysis   RoBERTa Analysis
 (NLTK)           (Hugging Face)
       |             |
       v             v
 Lexicon-Based    Transformer-Based
 Sentiment Scores Sentiment Probabilities
       |             |
       +------+------+
              |
              v
    Sentiment Comparison
              |
              v
 Visualizations and Review Examples
              |
              v
   amazon_food_sentiment_results.csv
```

## 3. Main Components

### Data Input
The system reads the Amazon Fine Food Reviews CSV dataset containing review text, star ratings, and available review metadata.

### Data Preprocessing
Pandas is used to validate columns, remove invalid records, clean review text, and select a reproducible sample of up to 2,000 reviews.

### Exploratory Data Analysis
Matplotlib and Seaborn generate charts showing the star-rating distribution and review lengths.

### VADER Sentiment Analysis
NLTK's VADER sentiment analyser calculates negative, neutral, positive, and compound sentiment scores.

### RoBERTa Sentiment Analysis
A pretrained Cardiff NLP RoBERTa model estimates negative, neutral, and positive sentiment probabilities for each review.

### Results and Visualization
The model outputs are combined with review metadata, compared across star ratings, visualized, and exported to a CSV file.

## 4. Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- NLTK and VADER
- Hugging Face Transformers
- PyTorch
- SciPy

## 5. Output

The primary output is `amazon_food_sentiment_results.csv`, containing processed review data and sentiment scores from both approaches.

The system also generates visualizations and prints examples of reviews whose textual sentiment differs from their star ratings.
