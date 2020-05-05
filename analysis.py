#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:32:51 2019

@author: xxxxx
"""

#python3
def get_content(soup_result):
    #soup result is a soup result object
    #method to return content attribute from bracket
    step1 = str(soup_result).split('content="')
    step2 = step1[1].split('" ')
    return step2[0]

def parse_text(obj, data):
    #obj is a soup result object
    #data is a list of dictionaries
    #method to parse object and place text associated with html meta data
    #including special tags into last dictionary in the list
    if obj.text == "":
        return
    full_text = obj.text.replace('\xa0', '')
    tags = ['a', 'strong']
    types = {'a':'link', 'strong':'bold'}
    for tag in tags:
        parse = obj.find_all(tag)
        data[-1][types[tag]].extend([item.text for item in parse])
        for remove_text in data[-1][types[tag]]: #step to remove special tag text from full_text
            full_text = full_text.replace(remove_text, '')
    #includes puncuation in common words
#    common_words = ['is', 'the', 'a', 'an', 'to', 'when', 'where', 'how', 'why',
#                    '!', '.', '?', 'thing', 'anything', 'including', 'his', 'her', 
#                    'it', "it", 'who', "'s", 'on ', 'in ', 'of ', 'for ']
#    for word in common_words:
#        full_text.replace(word, '')
#did not work out as expected maybe can use somewhere else
    if full_text == '': 
        return
    data[-1]['text'].append(full_text)
#noticed this approach to parsing the full_text down to the normal text causes 
#words to sometimes combine. 


import pandas as pd
json_data = pd.read_json('hs_2015_2016.json')
#list used for proof of concept to make sure no comments were present in html-doms
#l = list()
#x = list()
from bs4 import BeautifulSoup
data = []
i = 1
for html in json_data.itertuples():
    data.append({})
    html_string = html._1
    soup = BeautifulSoup(html_string, 'html.parser')
    web_name = get_content(soup.find('meta', attrs={'name':'original-source'}))
    data[-1]['url'] = web_name
    content = soup.find('div', attrs={'class':'entry-content', 'itemprop':'text'})
    split_content = content.find_all('p') #does not include lists
    data[-1]['link'] = list() #corresponds to a tag
    data[-1]['text'] = list() #no tag corresponds with this
    data[-1]['bold'] = list() #corresponds to strong tag
    for line in split_content:
        parse_text(line, data)
    data[-1]['list'] = [item.text for item in content.find_all('li')] #corresponds to li tag
    i += 1
    if i % 5 == 0:
        print("on step %d" %i)

#html string processing used for proof of concept to make sure no comments were present in html-doms
#    s1 = html_string.split('<div class="comment-respond" id="respond">')
#    try:
#        s2 = s1[1].split('<form')
#        l.append(s2[0])
#        x.append(s2[0].split('</h3>')[1])
#    except IndexError: continue

#building word cloud
#import matplotlib.pyplot as pPlot
from wordcloud import WordCloud, STOPWORDS
import numpy as npy
from PIL import Image

def create_word_cloud(line, image_name):
   maskArray = npy.array(Image.open("clouds-5b6b4e50c9e77c0050491212.jpg"))
   cloud = WordCloud(background_color = "white", max_words = 100, mask = maskArray, stopwords = set(STOPWORDS))
   cloud.generate(line)
   cloud.to_file(image_name)
review_text = " ".join(item for row in data for item in row['text']) 
review_lists = " ".join(item for row in data for item in row['list'])
create_word_cloud(review_text, 'text_cloud.jpg')
create_word_cloud(review_lists, 'list_cloud.jpg')
    

#cleaning, tokenizing and parsing text
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

def emo_parse(pre_key, emo_test, data_row):
    #parse through emo_test data
    #append pre_key to emo_test dictionary key and add emo_test data to data_row
    #finally add overall sentiment to data_row
    for key in emo_test:
        data_row[pre_key + "_" + key] = emo_test[key]
    if emo_test['compound'] >= .05:
        data_row[pre_key + "_" + 'sentiment'] = 'positive'
    elif emo_test['compound'] <= -.05:
        data_row[pre_key + "_" + 'sentiment'] = 'negative'
    else:
        data_row[pre_key + "_" + 'sentiment'] = 'neutral'

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
# clean text data
for row in data:
    row['clean_text'] = " ".join(clean_text(item) for item in row['text'])
    row['clean_list'] = " ".join(clean_text(item) for item in row['list'])
    #taking cleaned tokenized text and trying to analyze the emotion
    emo_test_text = sid.polarity_scores(row['clean_text'])
    emo_test_list = sid.polarity_scores(row['clean_list'])
    #placing analyzed emotion in data row
    emo_parse('text', emo_test_text, row)
    emo_parse('list', emo_test_list, row)

emo_data_set = pd.DataFrame(data)

#lets test our observation from word cloud on the text sentiment
#note this is bugged and keeps placing a None above positive
pie = emo_data_set.groupby('text_sentiment').size().plot(kind='pie', title='text sentiment')
fig = pie.get_figure()
fig.savefig('text_sentiment_pie.jpg')

pie = emo_data_set.groupby('list_sentiment').size().plot(kind='pie', title='list sentiment')
fig = pie.get_figure()
fig.savefig('list_sentiment_pie.jpg')

#create word cloud for only negative_text sentiment
negative_text = ' '.join(review for review in emo_data_set.loc[emo_data_set['text_sentiment']=='negative','clean_text'])
create_word_cloud(negative_text, 'negative_reviews.jpg')
#skipping list_sentiment because the majority is neutral and few negatives

emo_data_set = emo_data_set.assign(negative = emo_data_set['text_sentiment'] == 'negative')
def calc_neg(x):
    if x:
        return 1
    else:
        return 0
emo_data_set['negative'] = emo_data_set['negative'].map(calc_neg)
emo_data_set[['url', 'negative']].to_csv('url_summary.csv', index=False)