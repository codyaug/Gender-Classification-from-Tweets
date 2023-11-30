# Gender Classification from Tweets

This project aims to build a classifier that distinguishes between male and female Twitter users based on the content of their tweets. The dataset includes tweets from 200 individuals, half annotated as male and half as female.

## Model Choice and Approach

The choice of the **Naive Bayes Classifier** for this gender classification task and the overall approach were driven by several considerations:

### Naive Bayes Classifier:
The Naive Bayes Classifier is a probabilistic model based on Bayes' theorem, particularly suitable for text classification tasks. Here's why it was chosen:

- **Efficiency:** Naive Bayes models are computationally efficient, making them well-suited for quick exploration and prototyping within a limited time frame (4 hours in this case).
  
- **Text Classification:** Naive Bayes is widely used in text classification tasks, such as spam detection and sentiment analysis. Its simplicity and effectiveness in handling high-dimensional data, like the TF-IDF features derived from tweets, made it a natural choice.

- **Baseline Model:** Naive Bayes provides a solid baseline model for gender classification. Its performance, while not always the most advanced, serves as a reference point for evaluating more complex models in future iterations.

### Chosen Approach:

The project followed a structured approach encompassing data exploration, preprocessing, feature extraction, visualization, model training, and feature importance analysis. Here's the rationale behind this approach:

- **Holistic Understanding:** The step-by-step approach ensures a holistic understanding of the data and model behavior, making it accessible for both technical and non-technical audiences.

- **Feature Importance Analysis:** Including feature importance analysis helps interpret the model and provides insights into the linguistic and non-linguistic features contributing to gender classification.

- **Time Constraint:** Given the 4-hour time constraint, the approach prioritized clarity, efficiency, and key insights over complexity.

This combination of the Naive Bayes model and the structured approach aims to strike a balance between simplicity, interpretability, and effectiveness within the project's scope.

Feel free to explore alternative models and approaches based on the specific requirements of your gender classification task and the characteristics of the dataset.

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



-**Explore the Results:
Review the generated visualizations, model performance metrics, and feature importance analysis.

-**Key Visualizations
Distribution of Number of Words in Tweets
Distribution of Number of Words

-**Scatter Plot of Number of Words vs. Average Word Length
Scatter Plot

-**Correlation Matrix of Metafeatures
Correlation Matrix

**Conclusion
This project provides insights into the characteristics of tweets and how they relate to gender classification. Further improvements could include additional data exploration and model tuning.
