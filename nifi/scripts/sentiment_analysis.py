#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:18:33 2023

@author: 13ehrad
"""

#Load the libraries
import sys
import io
# import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from wordcloud import WordCloud,STOPWORDS
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
# import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
# from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
# from textblob import TextBlob
# from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# import os
# print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')

import traceback

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text



if __name__ == "__main__":
    
    # Read input CSV data from stdin
    input_data = sys.stdin.read()

    # Process CSV data using pandas
    imdb_data = pd.read_csv(io.StringIO(input_data))
    # imdb_data = pd.read_csv("/Users/13ehrad/Documents/Lancaster_University/Applied_Data_Mining/Data_Engineering/cw2/413_boilerplate/nifi/datasets/IMDB_Dataset.csv")
    
    #split the dataset  
    #train dataset
    train_reviews=imdb_data.review[:40000]
    train_sentiments=imdb_data.sentiment[:40000]
    #test dataset
    test_reviews=imdb_data.review[40000:]
    test_sentiments=imdb_data.sentiment[40000:]
    
    # print(test_reviews[40000])
    
    #Tokenization of text
    tokenizer=ToktokTokenizer()
    #Setting English stopwords
    stopword_list=nltk.corpus.stopwords.words('english')
    
    #Apply function on review column
    imdb_data['review']=imdb_data['review'].apply(denoise_text)
    
    #Apply function on review column
    imdb_data['review']=imdb_data['review'].apply(remove_special_characters)
    
    #Apply function on review column
    imdb_data['review']=imdb_data['review'].apply(simple_stemmer)
    
    #set stopwords to english
    stop=set(stopwords.words('english'))
    #Apply function on review column
    imdb_data['review']=imdb_data['review'].apply(remove_stopwords)
    
    #normalized train reviews
    norm_train_reviews=imdb_data.review[:40000]
    #Normalized test reviews
    norm_test_reviews=imdb_data.review[40000:]
    
    #Count vectorizer for bag of words
    cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,2))
    #transformed train reviews
    cv_train_reviews=cv.fit_transform(norm_train_reviews)
    #transformed test reviews
    cv_test_reviews=cv.transform(norm_test_reviews)
    
    # #Tfidf vectorizer
    # tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
    # #transformed train reviews
    # tv_train_reviews=tv.fit_transform(norm_train_reviews)
    # #transformed test reviews
    # tv_test_reviews=tv.transform(norm_test_reviews)
    
    
    #labeling the sentient data
    lb=LabelBinarizer()
    #transformed sentiment data
    sentiment_data=lb.fit_transform(imdb_data['sentiment'])
    
    #Spliting the sentiment data
    train_sentiments=sentiment_data[:40000]
    test_sentiments=sentiment_data[40000:]
    # -----------------------------------------------------------
    try:
        #training the model
        lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
        # #Fitting the model for Bag of words
        lr_bow=lr.fit(cv_train_reviews,train_sentiments)
        # lr_bow=lr.fit(cv_test_reviews,test_sentiments)
    except Exception as e:
        print(f"An error occurred during LogisticRegression fit operation: {e}")
        print(traceback.format_exc())
    #Fitting the model for tfidf features
    # lr_tfidf=lr.fit(tv_train_reviews,train_sentiments)
    # print(f"Type and shape of cv_train_reviews: {type(cv_train_reviews)}, {cv_train_reviews.shape}")
    # print(f"Type and shape of train_sentiments: {type(train_sentiments)}, {train_sentiments.shape}")

    
    #Predicting the model for bag of words
    lr_bow_predict=lr.predict(cv_test_reviews)
    # ##Predicting the model for tfidf features
    # # lr_tfidf_predict=lr.predict(tv_test_reviews)
    
    #Accuracy score for bag of words
    lr_bow_score=accuracy_score(test_sentiments,lr_bow_predict)
    
    #Accuracy score for tfidf features
    # lr_tfidf_score=accuracy_score(test_sentiments,lr_tfidf_predict)
    
    # print(lr_bow_score, lr_tfidf_score)
    
    clf_prediction =  {'Review' : test_reviews, 'Prediction' : lr_bow_predict}
    clf_prediction = pd.DataFrame(clf_prediction)
    
    # # Add your logic to infer the new column value for each row
    # df["new_column"] = df.apply(infer_new_column_value, axis=1)

    # Write the updated CSV data to stdout
    clf_prediction.to_csv(sys.stdout, index=False)
    

# import pandas as pd
# import sys
# import io

# def infer_new_column_value(row):
#     return "value_based_on_logic"

# if __name__ == "__main__":
#     # Read input CSV data from stdin
#     input_data = sys.stdin.read()

#     # Process CSV data using pandas
#     df = pd.read_csv(io.StringIO(input_data))

#     # Add your logic to infer the new column value for each row
#     df["new_column"] = df.apply(infer_new_column_value, axis=1)

#     # Write the updated CSV data to stdout
#     df.to_csv(sys.stdout, index=False)

