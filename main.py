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
#nltk.download('stopwords')
#nltk.download('punkt')
"""
   Start ML with 10th period
   Hacer en otro código :
   -Obtener Término frecuencia y TF IDF con los nuevos campos
   -Obtener top 100, top 50

"""
def main():
    print('Machine learning program.')
    print('1.Count words')
    print('2.Build a matrix')
    op=input()
    op=int(op)

    cloud_config= {

    'secure_connect_bundle': pathtohere+'//secure-connect-dbquart.zip'
         
    }
    
    objCC=CassandraConnection()
    auth_provider = PlainTextAuthProvider(objCC.cc_user,objCC.cc_pwd)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()
    querySt="select * from thesis.tbthesis where period_number=10 ALLOW FILTERING"   
        
    row=''
    

    if op==1:   
        print('1.Graph')
        print('2.WordCloud')
        op_1=input()
        op_1=int(op_1)   

        ltDoc=StringIO()
    
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
            #ltDocuments.append(thesis)
            ltDoc.write(thesis+' ')
            
            
        print('Cleaning data...')
        #Complete words 
        words=word_tokenize(ltDoc.getvalue(),language='spanish')
        #Words without punctuation
        words_no_pun=[]
        for w in words:
            if w.isalpha():
                words_no_pun.append(w.lower())
            #Remove stopwords
        sw=stopwords.words('spanish')
        clean_words=[]
        for w in words_no_pun:
            if w not in sw:
                clean_words.append(w)
        
    
        if op_1==1:

            #Clean words
            fdist=FreqDist(clean_words)
            fdist.plot(30)

        if op_1==2: 
            print('World Cloud process...')
            fdist=FreqDist(clean_words)
            wordcloud=WordCloud().generate_from_frequencies(fdist)
            plt.figure(figsize=(12,12))
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.show()
        

    if op==2:
        print('Building matriz')
        print('1.Vectorize')
        print('2.Get TF-IDF')
        op_1=input()
        op_1=int(op_1)  

        
        ltDocuments=[]
        #Read and deliver a list of documents
        statement = SimpleStatement(querySt, fetch_size=1000)
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

        sw=stopwords.words('spanish')    
        cv=CountVectorizer(encoding='utf-8',stop_words=sw)
        #Here in 'df' , we have all the information into a matriz of rows(thesis) and columns (terms)
        
            
        if op_1==1:
            print('Vectorizing...')
            print('--Información de la matriz (rows,cols)--')
            vector=cv.fit_transform(ltDocuments)
            dataFrame=pd.DataFrame()
            dataFrame=vector
            print(dataFrame.shape)
           
            

        if op_1==2:
            print('TF-IDF...') 
            #Process to get the TF-IDF
            tfidf_vectorizer=TfidfVectorizer(use_idf=True)
            tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(ltDocuments) 
            print(tfidf_vectorizer_vectors)
           
            
            

    print('...End...')   
    
    
     
        
    
class CassandraConnection():
    cc_user='quartadmin'
    cc_pwd='P@ssw0rd33'


if __name__=='__main__':
    main()