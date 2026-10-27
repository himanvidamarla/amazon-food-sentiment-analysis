# amazon-food-sentiment-analysis
Amazon Fine Food Reviews Sentiment Analysis using VADER and RoBERTa.


## 1. Project Overview

This project performs sentiment analysis on Amazon Fine Food Reviews using Natural Language Processing (NLP). It analyses customer reviews and compares two sentiment analysis approaches: VADER, a lexicon-based sentiment analyser, and RoBERTa, a pretrained transformer-based language model.

The objective is to understand customer opinions, explore the relationship between review text and star ratings, visualize sentiment patterns, and identify reviews where textual sentiment differs from the assigned rating.

## 2. Objectives

- Analyse customer reviews using NLP techniques.
- Explore the distribution of review star ratings.
- Examine review lengths and textual patterns.
- Calculate positive, neutral, and negative sentiment scores using VADER.
- Predict negative, neutral, and positive sentiment probabilities using RoBERTa.
- Compare sentiment patterns across one-star to five-star reviews.
- Identify reviews with inconsistencies between textual sentiment and star ratings.
- Export the analysis results for further investigation.

## 3. Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Natural Language Processing:** NLTK, VADER
- **Transformer Model:** Cardiff NLP Twitter RoBERTa Sentiment Model
- **Deep Learning Framework:** PyTorch
- **Model Utilities:** Hugging Face Transformers, SciPy
- **Development Environment:** Visual Studio Code

## 4. Dataset

The project uses the Amazon Fine Food Reviews dataset, which contains customer reviews of food products sold on Amazon.

Important fields used in this project include:

| Column      | Description                 |
| ----------- | --------------------------- |
| `Id`        | Unique review identifier    |
| `Score`     | Customer rating from 1 to 5 |
| `Text`      | Full review text            |
| `Summary`   | Short review summary        |
| `ProductId` | Product identifier          |
| `UserId`    | Reviewer identifier         |

The script analyses a reproducible random sample of up to 2,000 valid reviews using a fixed random seed of 42.

**Dataset source:** [Amazon Fine Food Reviews – Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

## 5. Methodology

### Step 1: Data Loading and Preprocessing

- Load the dataset using Pandas.
- Validate the required columns.
- Remove records with missing ratings or review text.
- Convert ratings to numeric values and retain valid ratings from 1 to 5.
- Remove empty review texts.
- Select up to 2,000 reviews using a fixed random seed.
- Calculate the number of words in each review.

### Step 2: Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the characteristics of the dataset.

The visualizations include:

- Distribution of reviews across one-star to five-star ratings.
- Distribution of review lengths based on word count.

### Step 3: VADER Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based sentiment analysis tool that estimates sentiment using words, punctuation, capitalization, and other textual cues.

It calculates four sentiment measures:

- `vader_neg`: Negative sentiment score.
- `vader_neu`: Neutral sentiment score.
- `vader_pos`: Positive sentiment score.
- `vader_compound`: Normalized overall sentiment score.

The compound score ranges from -1 to +1. Negative values indicate negative sentiment, positive values indicate positive sentiment, and values near zero indicate relatively neutral sentiment.

### Step 4: RoBERTa Sentiment Analysis

A pretrained RoBERTa model from Cardiff NLP is used to estimate the sentiment expressed in each review.

The model produces probabilities for three sentiment categories:

- **Negative:** `roberta_neg`
- **Neutral:** `roberta_neu`
- **Positive:** `roberta_pos`

These probabilities sum to approximately 1 for each successfully processed review.

The model is pretrained on Twitter text, so its performance on Amazon food reviews may differ from its performance in its original training domain.

### Step 5: Model Comparison

VADER and RoBERTa are compared using their sentiment scores across different star-rating categories.

The analysis includes:

- Average VADER compound scores by star rating.
- Average VADER positive scores by star rating.
- Average RoBERTa positive probabilities by star rating.
- Identification of reviews with unexpectedly positive sentiment despite a one-star rating.
- Identification of reviews with unexpectedly negative sentiment despite a five-star rating.

These comparisons help identify potential inconsistencies between the sentiment expressed in the review text and the assigned numerical rating.

## 6. Results and Observations

The implementation successfully processed **2,000 Amazon food reviews** and generated visualizations using exploratory data analysis, VADER, and RoBERTa.

The observed results include:

- **Rating distribution:** Five-star reviews were the most frequent category in the selected sample.
- **Review length:** The distribution was right-skewed, with most reviews being relatively short and a smaller number being considerably longer.
- **VADER analysis:** Average compound sentiment generally increased with star rating.
- **RoBERTa analysis:** Average positive sentiment probability also increased as star ratings rose from one to five stars.
- **Sentiment inconsistencies:** Some reviews contained sentiment that differed from the assigned star rating.

## 7. Output Files

### `amazon_food_sentiment_results.csv`

The primary output file contains the processed review data and sentiment scores generated by the analysis.

The output includes:

- Review metadata, including ID, star rating, review text, summary, product ID, and user ID where available.
- Review word counts.
- VADER negative, neutral, positive, and compound scores.
- RoBERTa negative, neutral, and positive sentiment probabilities.

The CSV file can be opened in Microsoft Excel or imported into Python for further analysis.

### Generated Visualizations

The script generates visualizations showing:

1. Number of reviews by star rating.
2. Review length distribution.
3. Average VADER compound score by star rating.
4. Average RoBERTa positive probability by star rating.
5. Comparison of VADER and RoBERTa positive sentiment scores across star ratings.

## 8. Installation and Execution

### Prerequisites

- The Amazon Fine Food Reviews dataset.

### Install Dependencies

Run the following command in the terminal:

```bash
pip install pandas numpy matplotlib seaborn nltk tqdm transformers torch scipy
```

###

### Run the Program

Open the terminal in the project directory and execute:

```bash
python senti-code.py
```

The program loads and preprocesses the dataset, generates visualizations, performs VADER and RoBERTa sentiment analysis, prints selected review examples, and exports the results to a CSV file.

**Note:** RoBERTa may take several minutes to process 2,000 reviews, depending on the computer's hardware. The pretrained model is downloaded the first time it is used and cached for subsequent runs.

## 9. Limitations

- The analysis uses a sample of up to 2,000 reviews rather than the complete dataset.
- The distribution of star ratings is imbalanced, which affects comparisons across rating categories.
- Star ratings do not always accurately represent the sentiment expressed in review text.
- RoBERTa was pretrained on Twitter text rather than specifically on Amazon food reviews.
- VADER relies on a sentiment lexicon and may struggle with complex context, sarcasm, and domain-specific expressions.
- The project compares sentiment scores and probabilities but does not establish model accuracy against an independently annotated sentiment dataset.
- Model predictions may be affected by negation, sarcasm, mixed opinions, and context-dependent language.
- Results from the selected sample may not represent the entire Amazon Fine Food Reviews dataset.

## 10. Conclusion

This project demonstrates the application of Natural Language Processing and sentiment analysis to customer reviews.

By comparing VADER, a lexicon-based approach, with a pretrained RoBERTa transformer model, the project explores how different methods estimate sentiment from textual customer feedback.

The findings show a general relationship between customer ratings and predicted positive sentiment. However, individual reviews also reveal inconsistencies between the sentiment expressed in the text and the numerical rating assigned by the customer.
