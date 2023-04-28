# -*- coding: utf-8 -*-
"""CIS 5450 Final Project - Sentiment Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G3d0maJZHNQ1od6_ekc4wZle-Bk24GHe

**Step 1**: Import necessary packages and set-up environment.
"""

# Install Flair for sentiment analysis
!pip install flair

import pandas as pd
import flair
from google.colab import drive
drive.mount('/content/drive')

"""**Step 2**: Load data from Google Drive."""

# Check file list in current directory
!ls '/content/drive/My Drive'

# Load data from csv to pandas dataframe
path = '/content/drive/MyDrive/Colab Notebooks/wsb_post_filtered_by_tickers.csv'
wsb_data_df = pd.read_csv(path)

"""**Step 3**: Check characteristics of the dataset."""

# Check number of records loaded
len(wsb_data_df)

# Check data type
wsb_data_df.dtypes

# Print data in first 5 rows
wsb_data_df.head(5)

"""**Step 4**: Load the pre-trained `Flair` model and tokenizer"""

flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

"""**Step 5**: Define function to extract the sentiment score. Since `Flair` will carry out label of the sentence in `POSITIVE` and `NEGATIVE`, we will combine the infomration of the label with the score predicted by this model."""

# Define the function to extract sentiment score
def senti_score(n):
    s = flair.data.Sentence(n)
    flair_sentiment.predict(s)
    total_sentiment = s.labels[0]
    assert total_sentiment.value in ['POSITIVE', 'NEGATIVE']
    sign = 1 if total_sentiment.value == 'POSITIVE' else -1
    score = total_sentiment.score
    return sign * score

"""**Step 6**: Apply the function to `wsb_data_df` and collect the outputs."""

wsb_data_df['sentiment'] = wsb_data_df.comment.map(senti_score)

"""**Step 7**: Export the dataframe as **CSV** file."""

wsb_data_df.to_csv(r'/content/drive/MyDrive/Colab Notebooks/wsb_data_with_senti.csv', index=False)