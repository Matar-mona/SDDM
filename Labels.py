import wikipedia as wiki
import nltk
from nltk.util import ngrams
import numpy as np
import re
import urllib.parse
from collections import Counter
import operator
import requests
import pprint as pp

"""
####################################### READ ME#######################################################################

Below code predicts the label for a topic after genetrating the Topic Model using LDA/NMF/Genism: 

How it works: 
1. It takes the topic model above, searchs for candidate labels using Wikipedia and return the top 8 labels which 
   serves as the primary candidates for labeling. 
2. We then create bigrams of these candidate labels and using post-tags we search through Wikipedia again to create 
   secondary labels for them. Most of these bigrams are just noise and we eliminate most of them out by using RACO(
   Related Article Conceptual Overlap) and Dice's coefficient to calculate the average intersection between two labels 
   in the same category. 
3. We then assign labels to tweets by selecting features which best describe the tweets really well. This is done by 
   searching for most frequent words which occurs in topic labels and averaging them and returning the max label.


   NB: I realized most of the bigrams were returning null values so I decided to comment them out and just focused on the 
   labels returned. Feel free to go ahead and mess around with the code.

   signed - Jeremy Johnson
########################################################################################################################
"""


class Labels:

    def __init__(self, terms_to_wiki):
        self.term_to_wiki = terms_to_wiki

    def return_dict_wiki_topics(self, **kwargs):
        return kwargs

    def return_topic_words_from_model(self, MyList=[], *args):
        return MyList

    #Generating candidate primary labels from wikipedia
    def get_titles_wiki(self):
        wiki_titles = {}
        for i in self.term_to_wiki:
            for k, v in i.items():
                s = ' '.join(v)
                results = wiki.search(s)
                results = results[:8]
                if results != None:
                    if k in wiki_titles:
                        wiki_titles[k].append(results)
                    else:
                        wiki_titles[k] = results
        j = dict([(str(k), v) for k, v in wiki_titles.items() if len(v) > 0])
        return j

    # Removing all null returned labels and returning only the topics with labels
    def remove_all_null_dicts_returned_from_wiki(self, **kwargs):
        new_keys_for_topic = {}
        words = self.return_topic_words_from_model(self.term_to_wiki)
        for k, v in kwargs.items():
            for i in words:
                for l in i.keys():
                    if int(k) == l:
                        new_keys_for_topic[k] = (v, i[l])

        return new_keys_for_topic

    #Calculating the frequency of the top words in the candidate labels
    def calculating_word_frequency(self, **kwargs):
        counter = Counter()
        frequent_words = dict()
        keys = {}
        l = []
        try:
            for k, v in kwargs.items():
                for i in v[0]:
                    data = urllib.parse.quote_plus(i)
                    data = re.sub(r'\+', '_', data)
                    URL = "https://en.wikipedia.org/wiki/" + data
                    with urllib.request.urlopen(URL) as source:
                        for line in source:
                            words = re.split(r"[^A-Z]+", line.decode('utf-8'), flags=re.I)
                            counter.update(words)
                        for word in kwargs[k][1]:
                            if i in frequent_words:
                                frequent_words[i].append((word, counter[word]))
                            else:
                                frequent_words[i] = [(word, counter[word])]

            for k, v in kwargs.items():
                for i in v[0]:
                    for j in frequent_words.keys():
                        if i == j:
                            if k in keys:
                                keys[k].append((j, frequent_words[j]))
                            else:
                                keys[k] = [(j, frequent_words[j])]

            # print(keys)
        except Exception as e:
            print(e)

        return keys

    #Predicting final label
    def predicting_label(self, **kwargs):
        bigger = dict()
        sum = 0
        for k, v in kwargs.items():
            for i in v:
                for j in i[1]:
                    sum += j[1]
                mean = sum / len(v)
                if k in bigger:
                    bigger[k].append((i[0], mean))
                else:
                    bigger[k] = [(i[0], mean)]


        results = []
        for k, v in bigger.items():
            counter = 0
            title = ''
            for i in v:
                if i[1] > counter:
                    counter = i[1]
                    title = i[0]
            results.append((k, title, counter))

        return results



