#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Statements
import json
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from palettable.colorbrewer.qualitative import Pastel1_7
from PIL import Image


# In[2]:


# Load user metadata from JSON file
user_data = []
with open('C:\\Users\\Dovid Glassner\\Downloads\\JN10004 -Narratize Data Scientist - Tech Interview Data Folder\\ds_interview\\manifest.jl', 'r') as json_file:
    for line in json_file:
        user_data.append(json.loads(line))
        # Check the number of users in the dataset
print(f"Number of users: {len(user_data)}")


# In[3]:


# Check the distribution of genders among users
gender_counts = pd.Series(user_data).value_counts('gender_human')
print(gender_counts)

# Visualize the gender distribution using a pie chart
gender_counts.plot.pie(autopct="%1.0f%%", labels=gender_counts.index, title="Gender Distribution")
plt.show()

# Check the number of users in the dataset
print(f"Number of users: {len(user_data)}")


# In[4]:


# Data Preprocessing
# Load tweets for each user
tweet_data = {}
for filename in os.listdir('C:\\Users\\Dovid Glassner\\Downloads\\JN10004 -Narratize Data Scientist - Tech Interview Data Folder\\ds_interview\\tweet_files'):
    user_id = filename.split('.')[0]
    with open(f'C:\\Users\\Dovid Glassner\\Downloads\\JN10004 -Narratize Data Scientist - Tech Interview Data Folder\\ds_interview\\tweet_files/{filename}', 'r', encoding='utf-8') as f:
        tweets = f.readlines()
    tweet_data[user_id] = tweets

# Combine tweets for each user
user_tweets = {}
for user_id, tweets in tweet_data.items():
    user_tweets[user_id] = ' '.join(tweets)

# Text Preprocessing
# Remove punctuation and convert text to lowercase
for user_id, text in user_tweets.items():
    user_tweets[user_id] = re.sub(r'[^\w\s]', '', text).lower()

# Remove stop words
stop_words = set(stopwords.words('english'))
for user_id, text in user_tweets.items():
    user_tweets[user_id] = ' '.join([word for word in text.split() if word not in stop_words])


# In[5]:


# Feature Extraction
# Create a DataFrame with user IDs, tweets, and gender labels
user_ids = [user.get('user_id_str') or user.get('id') for user in user_data]
tweets = list(user_tweets.values())  # Convert values to list
genders = [user.get('gender_human') or 'Unknown' for user in user_data]

data = pd.DataFrame({'user_id': user_ids, 'tweets': tweets, 'gender': genders})

# Extract TF-IDF features from tweets
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['tweets'])

# Explain dataset structure
print("\nDataset Structure Explanation:")
print("-------------------------------")
print("1. 'user_id_str': User ID")
print("2. 'num_tweets': Number of tweets by the user")
print("3. 'gender_human': Gender of the user")
print("-------------------------------")


# In[6]:


# Generate and analyze metafeatures
# Add metafeatures to the dataset
data['num_words'] = data['tweets'].apply(lambda x: len(x.split()))
data['num_unique_words'] = data['tweets'].apply(lambda x: len(set(x.split())))
data['avg_word_length'] = data['tweets'].apply(lambda x: np.mean([len(word) for word in x.split()]))

# Display summary statistics of metafeatures
print("\nSummary Statistics of Metafeatures:")
print(data[['num_words', 'num_unique_words', 'avg_word_length']].describe())


# In[7]:


# Visualize the dataset extensively using Matplotlib, seaborn, and Plotly
# Visualize the distribution of the number of words in tweets
plt.figure(figsize=(10, 6))
sns.histplot(data['num_words'], bins=30, kde=True)
plt.title('Distribution of Number of Words in Tweets')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()


# In[8]:


# Visualize the relationship between number of words and average word length
sns.scatterplot(x='num_words', y='avg_word_length', hue='gender', data=data)
plt.title('Relationship between Number of Words and Average Word Length in Tweets')
plt.xlabel('Number of Words')
plt.ylabel('Average Word Length')
plt.show()


# In[9]:


# Use Plotly to create an interactive scatter plot
fig = px.scatter(data, x='num_words', y='avg_word_length', color='gender', title='Interactive Scatter Plot',
                 labels={'num_words': 'Number of Words', 'avg_word_length': 'Average Word Length'})
fig.show()


# In[10]:


# Additional Visualizations
# Correlation Matrix
correlation_matrix = data[['num_words', 'num_unique_words', 'avg_word_length']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Metafeatures')
plt.show()


# In[11]:


# Pairplot for selected features
sns.pairplot(data[['num_words', 'num_unique_words', 'avg_word_length', 'gender']], hue='gender')
plt.suptitle('Pairplot of Metafeatures by Gender', y=1.02)
plt.show()


# In[12]:


# Boxplot of word count by gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='num_words', data=data)
plt.title('Boxplot of Number of Words in Tweets by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Words')
plt.show()


# In[13]:


# Violinplot of word length by gender
plt.figure(figsize=(10, 6))
sns.violinplot(x='gender', y='avg_word_length', data=data)
plt.title('Violinplot of Average Word Length in Tweets by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Word Length')
plt.show()


# In[22]:


# **Model Training and Evaluation**

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, data['gender'], test_size=0.2, random_state=42)

# Initialize the SGDClassifier with a sparse representation
model = SGDClassifier()

# Incremental training
batch_size = 100  # Adjust the batch size as needed
for i in range(0, X_train.shape[0], batch_size):
    X_batch = X_train[i:i+batch_size].toarray()
    y_batch = y_train[i:i+batch_size]
    model.partial_fit(X_batch, y_batch, classes=np.unique(y_train))

# Evaluate Model Performance
X_test = X_test.toarray()  # Convert sparse matrix to dense array
y_pred = model.predict(X_test)

# Use zero_division='warn' in classification_report
print(classification_report(y_test, y_pred, zero_division='warn'))


# In[25]:


# Get feature importances from the trained model
coef = model.coef_.ravel()
important_features = pd.DataFrame({'feature': vectorizer.get_feature_names_out(), 'importance': coef})

# Sort features by importance
important_features.sort_values(by='importance', ascending=False, inplace=True)

# Visualize Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=important_features.head(10))
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Most Important Features')
plt.show()

