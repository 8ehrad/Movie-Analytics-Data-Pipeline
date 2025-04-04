#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:56:26 2023

@author: 13ehrad
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import sys
import json
# nltk.download('vader_lexicon')

def scrape_movie_reviews(imdb_id):
    base_url = f"https://www.imdb.com/title/{imdb_id}/reviews/_ajax?ref_=undefined&paginationKey="
    reviews = []
    pagination_key = ""

    while True:
        url = base_url + pagination_key
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        review_divs = soup.find_all("div", class_="review-container")

        for review_div in review_divs:
            review = {}
            review["title"] = review_div.find("a", class_="title").text.strip()
            review["content"] = review_div.find("div", class_="text").text.strip()
            reviews.append(review)

        load_more_div = soup.find("div", class_="load-more-data")
        if load_more_div:
            pagination_key = load_more_div["data-key"]
        else:
            break

    return reviews


# def get_title(title):
    
#     api_key = 'f108fc59'
    
#     url = f'http://www.omdbapi.com/?t={title}&apikey={api_key}'
#     response = requests.get(url)
#     data = response.json()
    
#     if data['Response'] == 'True':
#         imdb_id = data['imdbID']
#         return imdb_id
#     else:
#         print("Movie not found")


def read_titles(filename):
    titles = pd.read_csv(filename, delimiter="\t")
    return titles 

if __name__ == "__main__":
    
    # imdb_id = sys.stdin.read()
    # title = 'The Matrix Resurrections'
    
    # imdb_id = get_title(title)
    
    # # Read input from stdin
    imdb_id = sys.stdin.read()
    # print(imdb_id)
    
    # Split input data and assign them to variables
    # imdb_id, title = input_data.strip().split('\n')
    
    sia = SentimentIntensityAnalyzer()
    
    reviews = scrape_movie_reviews(imdb_id)
        
    for i in range(len(reviews)):
        sentiment_dict = sia.polarity_scores(reviews[i]['content'])
    
        if sentiment_dict['compound'] >= 0.05 :
            overall_sentiment = "positive"
    
        elif sentiment_dict['compound'] <= - 0.05 :
            overall_sentiment = "negative"
    
        else :
            overall_sentiment = "neutral"
        
        reviews[i]['sentiment'] = overall_sentiment
        
    pos = len([item for item in reviews if item['sentiment']=='positive'])
    neg = len([item for item in reviews if item['sentiment']=='negative'])
    neu = len([item for item in reviews if item['sentiment']=='neutral'])
    
    # output = {'title' : title, 'positive' : pos, 'negative' : neg, 'neutral' : neu}
    output = {'positive' : pos, 'negative' : neg, 'neutral' : neu}
    
    # Write the output dictionary as a JSON string to stdout
    sys.stdout.write(json.dumps(output))

    
    


