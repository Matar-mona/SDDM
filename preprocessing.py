from Labels import Labels
from models import Models
import pandas as pd
import re
from nltk.corpus import stopwords
import string
import sqlite3
from time import time
import eventregistry as ER
import itertools
"""
 ############  READ ME ##############
Topic tree generation is in 2 parts:
1. We determine patterns in the users tweets using topic modeling 
2. Using disambiguity in wiki and the first two hierarchies in DMOZ to build the topic tree model deserved 
The code below is just implementation of the topic modeling and generating the patterns which is stored in csv file 
and later used in wiki and DMOZ. It takes only 3 mins to run on the 28 million rows 
We have 3 functions in the code: 
1. db() : this is a databsed used to stored large number of cleaned tweets since we cannot use pickle to serialize(only 
used on small datasets). We then read directly these values into pandas tables. The main reason for working with pandas
is to avoid loops which slows down the program.
2. create_dict() : transforms pandas table data into dictionary which is passed to the 3 function. It takes approximately
22 mins to execute. When done it stored in a pickle file which can be used instead of executing the whole process again.
3. creating_dataframe(): is a bad choice of function name but all is does is take the dictionary in step 2 apply 
all preprocessing steps(stopwords, stemming, etc) required for implementing the topic models. Then passed to the models 
to generate the patterns in self.userTweets. 
Two models are being useed here the LDA and NMF. I did this to check which one actually gives a good pattern recognition
to texts. The output is then read on to a csv file which can be used to building the topic tree
"""


class Preprocess_Main:

    allWordsFromUsers = []
    allWordsFromUsersJoined = []
    noneDuplicateWordsUsedFromAllUsers = []
    userTweets = []
    userTopicLabels = []
    userWordIndexes = []

    regex_str = [
        r'<[^>]+>',  # HTML tags
        r'(?:@[\w_]+)',  # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

        r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
        r'(?:[\w_]+)',  # other words
        r'(?:\S)'  # anything else
    ]

    regex_pat = [
        r'<[^>]+>',  # HTML tags
        r'(?:@[\w_]+)',  # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'  # URL
    ]

    tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
    replace_re = re.compile(r'(' + '|'.join(regex_pat) + ')', re.IGNORECASE)


    def __init__(self, df):
        self.df = df
        # pass
    @staticmethod
    def db(db):
        conn = sqlite3.connect(db)

        c = conn.cursor()

        # Create table
        # c.execute('''CREATE TABLE twitter (user_id text, tweets text)''')
        counter = 0
        # with open('/Users/jeremyjohnson/self.userTweets/gate.csv', encoding='ISO-8859-1') as f:
        #     cv = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        #     for i in cv:
        #         # if counter <= 10:
        #         p = ''.join(re.findall(r'\d{8} .+', ' '.join(i)))
        #         if p != '':
        #             t = (p.split()[0], (' '.join(p.split()[1:])))
        #             c.execute("INSERT INTO twitter VALUES (?,?)", t)
        #             counter +=1
        #             print(counter, t)
        #
        # print("Finished with DB")
        # conn.commit()
        # conn.close()

        # sql = "SELECT * FROM twitter"
        # t0 = time()
        # print('Executing command')
        # df = pd.read_sql(sql, conn)
        # print(df.head())
        # print("Executing time:", round(time() - t0, 3), "s")
        # conn.close()
        # print('### Starting to create dicts ###')
        # return df

    def create_dict(self):
        mydict = {}
        t0 = time()
        for i in range(len(self.df)):
            currentid = self.df.iloc[i, 0]
            currentvalue = self.df.iloc[i, 1]
            mydict.setdefault(currentid, [])
            mydict[currentid].append(currentvalue)
        print("### Executed time:", round(time() - t0, 3), "s ###")
        print('### Starting to create topics ###')
        return mydict

    @classmethod
    def tokenize(cls, s):
        return ' '.join(cls.tokens_re.findall(s))

    @classmethod
    def replace(cls, s):
        return re.sub(cls.replace_re, '', s)

    @classmethod
    def split(cls, s):
        return s.split()

    @classmethod
    def terms_only(cls, s):
        punctuation = list(string.punctuation)
        stop = stopwords.words('english') + punctuation + ['rt', 'via']
        return [item for item in s if item not in stop and not item.startswith(('#', '@'))]
    # @staticmethod
    def creating_dataframe(self, dictionary):
        final_words = []
        final_words1 = []
        
        l = []
        z = []
        docs ={}
        keys = dictionary.keys()
        for key in keys:
            kk = str(key)
            k = re.findall(r'\d{8}', kk)
            l.append(k)
        for i in l:
            for j in i:
                z.append(j)
        for key in z:
            # if key == '19234329':
            print("###################### Generating topic labels for {} ############################".format(key))
            df = pd.DataFrame(dictionary[key])
            df.columns = ['Text']
            df_ = df['Text'].apply(lambda x: ''.join(x))
            df_ = df_.str.lower()
            df_ = df_.apply(self.tokenize)
            df_ = df_.apply(self.replace)
            df_ = df_.apply(self.split)
            df_ = df_.apply(self.terms_only)
            df_ = df_.apply(lambda x: ' '.join(x))
            df_ = df_.apply(lambda x: re.sub(r' +', ' ', x))
            [final_words.append("".join(i).strip().split()) for i in df_]
            [final_words1.append(i) for i in final_words if len(i) >= 5]
            [self.userTweets.append(re.sub(r' +', " ", (' '.join(i)))) for i in final_words1]

            if key in docs:
                docs[key].append(self.userTweets)
            else:
                docs[key] = self.userTweets

            print(key,":",self.userTweets)
            currentWordsByUser = []
            for i in range(len(self.userTweets)):
                tweetWords = self.userTweets[i].strip("'")
                tweetWords = tweetWords.strip('"')
                tweetWords = tweetWords.strip(",")

                currentWordsByUser.append(list(set(str(tweetWords).split())))

            uniqueWordsByUser = list(set(list(itertools.chain.from_iterable(currentWordsByUser))))
            print("uniqueWordsByUser:",uniqueWordsByUser)
            print("len(uniqueWordsByUser):",len(uniqueWordsByUser))
            #append all unique words from each user to global word vector
            self.allWordsFromUsers.append(uniqueWordsByUser)

            ###
            
            mm = Models(50, 10, **docs)  #50,10
            terms_to_wiki = mm.calling_methods('LDA')
            ll = Labels(terms_to_wiki)
            wiki_titles = ll.get_titles_wiki()
            equal_length = ll.remove_all_null_dicts_returned_from_wiki(**wiki_titles)
            frq = ll.calculating_word_frequency(**equal_length)
            #print(equal_length)
            #print("------")
            #print(frq)

            results = ll.predicting_label(**frq)
            l = []
            for i in range(len(results)):
                er = ER.EventRegistry(apiKey='32db7607-6c90-40bd-b653-e167da1462c9')
                analytics = ER.Analytics(er)
                cat = analytics.categorize(results[i][1])
                for k, v in cat.items():
                    if k == 'categories':
                        for y, value in v[0].items():
                            if y == 'label':
                                l.append(value.split('/')[2])

            
            self.userTopicLabels.append(l)
            
        print('########### FINAL FILE EXECUTED ##################')  
        self.allWordsFromUsersJoined = list(itertools.chain.from_iterable(self.allWordsFromUsers))  #joined        
        self.noneDuplicateWordsUsedFromAllUsers = list(set(self.allWordsFromUsersJoined))
        self.allUsersIndexing()
        self.savePreprocessedData()


    def allUsersIndexing(self):
        for i in range(len(self.allWordsFromUsers)):
            wordIndexes = []
            for j in range(len(self.allWordsFromUsers[i])):
                currentWord = str(self.allWordsFromUsers[i][j])
                #print("currentWord:",currentWord)
                for k in range(len(self.noneDuplicateWordsUsedFromAllUsers)):
                    if currentWord == self.noneDuplicateWordsUsedFromAllUsers[k]:
                        wordIndexes.append(k)
            self.userWordIndexes.append(wordIndexes)
                    

    def savePreprocessedData(self):
        with open('preprocessedData.txt', 'w') as preprocessed:
            preprocessed.write(str(self.noneDuplicateWordsUsedFromAllUsers) + '\n' )
            for i in range(len(self.userTopicLabels)):
                preprocessed.write(str(self.userWordIndexes[i]) + ' ' + str(self.userTopicLabels[i]) + '\n' )