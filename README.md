# Sentiment-Analysis-for-Restaurant-Reviews
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg) ![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![NLTK](https://img.shields.io/badge/Library-NLTK-orange.svg)
• A glimpse of the web app:

![GIF](restaurant-review-web-app.gif)


• If you want to view the deployed model, then go to following links mention below:<br />
Web App Link: _https://sentimentanalysisforrestaurant.herokuapp.com/_<br />

# Introduction

- Automate detection of different sentiments from textual comments and feedback, A machine learning model is created to understand the sentiments of the restaurant reviews. The problem is that the review is in a textual form and the model should understand the sentiment of the review and automate a result. 
- The main motive behind this project is to classify whether the given feedback or review in textual context is positive or negative. 
Reviews can be given to the model and it classifies the review as a negative review or a positive. This shows the satisfaction of the customer or the experience the customer has experienced.
- The basic approach was trying a different machine learning model and look for the one who is performing better on that data set. The restaurant reviews are very related to the project topic as reviews are made on websites and we can apply this model on such data sets to get the sentiments.

# 2. Problem Definition and Algorithm

### 2.1 Task Definition

> To develop a machine learning model to detect different types of sentiments contained in a collection of English sentences or a large paragraph.

I have chosen Restaurant reviews as my topic. Thus, the objective of the model is to correctly identify the sentiments of the users by reviews which is an English paragraph and the result will be in positive or negative only.

For example, 

If the review given by the user is:
> “ We had lunch here a few times while on the island visiting family and friends. The servers here are just wonderful and have great memories it seems. We sat on the oceanfront patio and enjoyed the view with our delicious wine and lunch. Must try! ”

Then the model should detect that this is a positive review. Thus the output for this text will be **Positive**.

### 2.1 Algorithm Definition

The data set which I chose for this problem is available on Kaggle. The sentiment analysis is a classification because the output should be either positive or negative.I have used Multinomial Naive Bayes classification algorithms on this data set.

 Multinomial Naive Bayes:
Naive Bayes Classifier Algorithm is a family of probabilistic algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of a feature.
Bayes theorem calculates probability P(c|x) where c is the class of the possible outcomes and x is the given instance which has to be classified, representing some certain features.

> P(c|x) = P(x|c) * P(c) / P(x)

Naive Bayes is mostly used in natural language processing (NLP) problems. Naive Bayes predict the tag of a text. They calculate the probability of each tag for a given text and then output the tag with the highest one.


# Experimental Evaluation

### 3.1 Methodology

All the models were judged based on a few criteria. These criteria are also recommended by the scikit-learn website itself for the classification algorithms.
The criteria are:
* Accuracy score: 
Classification Accuracy is what we usually mean when we use the term accuracy. It is the ratio of the number of correct predictions to the total number of input samples.



* Confusion Matrix:
A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. 
i) There are two possible predicted classes: "yes" and "no". If we were predicting the presence of a disease, for example, "yes" would mean they have the disease, and "no" would mean they don't have the disease.
ii) The classifier made a total of 200 predictions (e.g., 200 patients were being tested for the presence of that disease).
iii) Out of those 200 cases, the classifier predicted "yes" 106 times, and "no" 94 times.
iv) In reality, 103 patients in the sample have the disease, and 97 patients do not.
	* true positives (TP): These are cases in which we predicted yes (they have the disease), and they do have the disease.
	* true negatives (TN): We predicted no, and they don't have the disease.
	* false positives (FP): We predicted yes, but they don't have the disease. (Also known as a "Type I error.")
	* false negatives (FN): We predicted no, but they do have the disease. (Also known as a "Type II error.")

* Precision: It is the number of correct positive results divided by the number of positive results predicted by the classifier.

* Recall: It is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive).

### 3.2 Result

The result of the evaluation of the metrics of the machine learning models is mentioned below:

i) Multinomial Naive Bayes:

* Confusion Matrix:
 [[ 72,  25],
 [ 22,   81]]

* Accuracy, Precision and Recall
	Accuracy is   78.5 % (with alpha value as 0.2)
            Precision is  0.76
            Recall is  0.79


# 4. Work

* The approach was straight forward. I have selected a few classifiers algorithms for my project. I chose Sentiment-Analysis-for-Restaurant-Reviews as my project title. Firstly I understood the working of the algorithm and read about them.

* After gathering the data set from Kaggle. The first step was to process the data. In data processing, I used NLTK (Natural Language Toolkit) and cleared the unwanted words in my vector. I accepted only alphabets and converted it into lower case and split it in a list.
Using the PorterStemmer method stem I shorten the lookup and Normalized the sentences.
Then stored those words which are not a stopword or any English punctuation. 

* Secondly, I used CountVectorizer for vectorization. Also used fit and transform to fit and transform the model. The maximum features were 1000.

* The next step was Training and Classification. Using train_test_split 20% of data was used for testing and remaining was used for training. The data were trained on the Multinomial Naive Baye algorithms mentioned above. 

* Later metrics like Confusion matrix, Accuracy, Precision, Recall were used to calculate the performance of the model.

* The best model was tuned to get a better result using hyperparameter tunning.

* Lastly, we checked the model with real reviews and found the model is detecting the sentiments of the customer reviews properly.


# 5. Heroku Deployment

Thanks to Krish Naik Sir Heroku deployment was very easy. I found his video and followed the steps. It would be more better to watch his video rather than explaining whole process and there are very few steps.

Here is the [link](https://youtu.be/mrExsjcvF4o) of the video.

# 6. Future Work

There is always a scope of improvement. Here are a few things which can be considered to improve. 
* Different classifier machine leaning models such as Bernoulli Naive Bayes,Logistic Regression can also be tested.
* Try a different data set. Sometimes a data set plays a crucial role too. 
* Some other tuning parameters to improve the accuracy of the model.

# 7. Conclusion

The motive of the model is to correctly detect the sentiments of the textual reviews or feedback. The developed model has an accuracy of 78.5% and successfully detects the sentiments of the textual reviews or feedback.
The model has been tested with few of the online reviews and was found that it detects the sentiments correctly.
Thus, can conclude that the motive was successful and the model can be used to detect the sentiments of the reviews and feedback.
