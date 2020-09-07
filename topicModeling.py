import os
import nltk
import pandas as pd
from io import StringIO
#sent or word tokenize: Get the information into sentences or words
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
from wordcloud import WordCloud
import matplotlib.pyplot as plt

pathtohere=os.getcwd()
nltk.download('stopwords')
nltk.download('punkt')





class CassandraConnection():
    cc_user='quartadmin'
    cc_pwd='P@ssw0rd33'


if __name__=='__main__':
    main()