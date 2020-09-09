from io import StringIO
import nltk
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import os
nltk.download('stopwords')
nltk.download('punkt')
pathtohere=os.getcwd()
sw=stopwords.words('spanish')


def convertListToString(lst):
    print('Converting Listo to String...')
    strDoc=StringIO()
    for doc in lst:
        strDoc.write(str(doc)+' ')
    return strDoc.getvalue()


def clean_corpus(words,sw):
    print('Cleaning corpus: Getting rid of puntuaction and stopwords...')
    words_no_pun=[]
    for w in words:
        if w.isalpha():
            words_no_pun.append(w.lower())
    #Remove stopwords
    clean_words=[]
    for w in words_no_pun:
        if w not in sw:
            clean_words.append(w)
    return clean_words    

def get_TFIDF():
    print('Getting TF-IDF matrix...')
    lsDocuments=[]
    lsDocuments=getCorpusList()
    tfidf=TfidfVectorizer(encoding='utf-8',stop_words=sw,smooth_idf=True)
    lsReturn=[]
    lsReturn.append(tfidf)
    lsReturn.append(ltDocuments)
    return lsReturn

def getCountVectorizer():
    
    lsDocuments=[]
    lsDocuments=getCorpusList()
    count_vect = CountVectorizer(stop_words=sw)
    lsReturn=[]
    lsReturn.append(count_vect)
    lsReturn.append(ltDocuments) 

    return lsReturn

def getCorpusList():
    print('Getting information from database into a python list...')
    cloud_config= {

    'secure_connect_bundle': pathtohere+'//secure-connect-dbquart.zip'
         
    }
    
    objCC=CassandraConnection()
    auth_provider = PlainTextAuthProvider(objCC.cc_user,objCC.cc_pwd)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()
    row=''
    ltDocuments=[]
    querySt="select heading,text_content,subject,type_of_thesis from thesis.tbthesis where period_number=10 ALLOW FILTERING"       
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

    return ltDocuments
        

   

    
    
class CassandraConnection():
    cc_user='quartadmin'
    cc_pwd='P@ssw0rd33'        