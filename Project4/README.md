# Project4

Run:

```
streamlit run app.py --server.port 8604
```

# Pipeline
1. If you have no csv files, run data_preprocessor.py to generate dataset and preprocess it for model training.
2. If you have no data visualizations (visualizations folder), run sentiment_score.py to vizualize the data and assign value to the words.

# Assignment Requirements
1. Preprocess and Visualize the Data:
a. Perform a descriptive statistical analysis of the data and decide how to handle missing values. (DONE)
b. Store your data in a dataframe.
c. Count the number of positive, negative, and neutral text items, as tagged by a score in one of the columns. (DONE)
d. Display your findings in a plot. (DONE)

2. Build the Model:
a. Remove punctuation! (DONE)
b. Remove stop words (i.e., words that do not add a sentiment). (DONE)
c. Assign each word in every text element, with a sentiment score (use TfidVectorizer). (DONE)
d. Use a binary classification algorithm (e.g., logistic regression), which you can import from sklearn.
e. Divide the data into a training set and testing set, with a ratio of 80:20.
f. Fit the data set using the model.
g. Compute the (accuracy) score of the model.
h. Make Predictions:
i. Enter several questions and assess the sentiment they convey.

3. Evaluate the Model:
a. Create a confusion matrix to assess the overall performance.
b. Present the performance metrics and visualize the findings.
c. Summarize the project, explaining to what extent it is suitable to perform sentiment analysis.