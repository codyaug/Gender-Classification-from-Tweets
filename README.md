# Gender Classification from Tweets

This project aims to build a classifier that distinguishes between male and female Twitter users based on the content of their tweets. The dataset includes tweets from 200 individuals, half annotated as male and half as female.

## Overview

The project follows these key steps:

1. **Data Loading and Exploration:**
   - Metadata for each user is stored in 'manifest.jl'.
   - Tweets for each user are stored in files corresponding to their Twitter ID.
   - Explore the dataset to understand its structure and distribution.

2. **Text Preprocessing:**
   - Clean and preprocess text data by removing punctuation, converting to lowercase, and eliminating stop words.

3. **Feature Extraction:**
   - Use TF-IDF (Term Frequency-Inverse Document Frequency) to extract linguistic features from tweets.
   - Include non-linguistic features like the number of words, unique words, and average word length.

4. **Data Visualization:**
   - Visualize the distribution of the number of words in tweets.
   - Explore relationships between features using scatter plots and interactive plots with Plotly.
   - Visualize correlations between different features.

5. **Model Training and Evaluation:**
   - Split data into training and testing sets.
   - Train a Naive Bayes Classifier.
   - Evaluate model performance using classification reports and confusion matrices.

6. **Feature Importance Analysis:**
   - Calculate mutual information between features and gender labels.
   - Visualize the top 10 most important features.



Explore the Results:
Review the generated visualizations, model performance metrics, and feature importance analysis.

Key Visualizations
Distribution of Number of Words in Tweets
Distribution of Number of Words

Scatter Plot of Number of Words vs. Average Word Length
Scatter Plot

Correlation Matrix of Metafeatures
Correlation Matrix

Conclusion
This project provides insights into the characteristics of tweets and how they relate to gender classification. Further improvements could include additional data exploration and model tuning.
