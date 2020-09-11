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
    print('Cleaning list of Documents: Getting rid of puntuaction and stopwords...')
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

def getRawTextToList():
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
    lsSubject=[]
    querySt="select heading,text_content,subject,type_of_thesis from thesis.tbthesis where period_number=10 ALLOW FILTERING"       
    statement = SimpleStatement(querySt, fetch_size=1000)
    print('Getting data from datastax...')
    for row in session.execute(statement):
        thesis_b=StringIO()
        #Add subject to a list aside
        lsSubject.append(row[2])
        for col in row:
            if type(col) is list:
                for e in col:
                    thesis_b.write(str(e)+' ')
            else:        
                thesis_b.write(str(col)+' ')
        thesis=''
        thesis=thesis_b.getvalue()
        ltDocuments.append(thesis)
    lsReturn=[]    
    lsReturn.append(ltDocuments)
    lsReturn.append(lsSubject)
    return lsReturn

def appendInfoToFile(path,filename,strcontent):
    txtFile=open(path+filename,'a+')
    txtFile.write(strcontent)
    txtFile.close()

def getDominantTopicDataFrame(lda_model,corpus):
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break


    # Add original text to the end of the output
    #'contents' is created as a new set of rows (Series) which can be added later 
    contents = pd.Series(lsDocuments_NoSW)
    subject=pd.Series(lsSubject)
    #concat([A,B]) is actually adding another column, hence is correct, so from 3 columns, it ends up with 4
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df = pd.concat([sent_topics_df, subject], axis=1)
    sent_topics_df.columns = ['Topic_No', 'Dominant_Topic', 'Keywords','Text_Without_stopwords','Subject'] 

    return sent_topics_df   
        

   

    
    
class CassandraConnection():
    cc_user='quartadmin'
    cc_pwd='P@ssw0rd33'        