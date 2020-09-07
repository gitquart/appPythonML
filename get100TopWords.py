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
import MLfunctions as mlf

pathtohere=os.getcwd()
nltk.download('stopwords')
nltk.download('punkt')

def main():
    print('Machine learning program.')
    cloud_config= {

    'secure_connect_bundle': pathtohere+'//secure-connect-dbquart.zip'
         
    }
    
    objCC=CassandraConnection()
    auth_provider = PlainTextAuthProvider(objCC.cc_user,objCC.cc_pwd)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()
    querySt="select heading,text_content,subject,type_of_thesis from thesis.tbthesis where period_number=10 ALLOW FILTERING"       
    row=''
    ltDocuments=[]
    #Read and deliver a list of documents
    statement = SimpleStatement(querySt, fetch_size=1000)
    print('Getting data from datastax...')
    for row in session.execute(statement):
        thesis_b=StringIO()
        for col in row:
            if type(col) is list:
                for e in col:
                    thesis_b.write(str(e)+' ')
            else:        
                thesis_b.write(str(col)+' ')
        thesis=''
        thesis=thesis_b.getvalue()
        ltDocuments.append(thesis)
        
    strDocuments=mlf.convertListToString(ltDocuments)    

    #Get the most common words (no TF IDF yet)
    sw=stopwords.words('spanish')
    words = word_tokenize(strDocuments,language='spanish')
    clean_words=mlf.clean_corpus(words,sw)

    fdist=FreqDist(clean_words)
    file = open(pathtohere+"\\100_TOP_WORDS_NO_TFIDF.txt","a+")
    for word,freq in fdist.most_common(100):
        file.write(word+' '+str(freq)+'\n') 

    """
    cv=CountVectorizer(stop_words=sw)
    word_count_vector=''
    word_count_vector=cv.fit_transform(ltDocuments)
    """


    
            

    
class CassandraConnection():
    cc_user='quartadmin'
    cc_pwd='P@ssw0rd33'


if __name__=='__main__':
    main()