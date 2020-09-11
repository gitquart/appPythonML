import MLfunctions as mlf
import gensim
from gensim import corpora
from pprint import pprint
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import pandas as pd
import os
sw=stopwords.words('spanish')
pathtohere=os.getcwd()


def main():
    print('LDA model with gensim, 1-gram')
    lsReturn=[]
    lsDocuments=[]
    lsSubject=[]
    #Get the the information into a list of documents
    lsReturn=mlf.getRawTextToList()
    lsDocuments=lsReturn[0]
    lsSubject=lsReturn[1]
    lsDocuments_NoSW = [[word for word in simple_preprocess(str(doc)) if word not in sw] for doc in lsDocuments]
    # Create Dictionary
    id2word = corpora.Dictionary(lsDocuments_NoSW)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in lsDocuments_NoSW]
    """
    
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                id2word=id2word,
                                num_topics=5, 
                                random_state=100)
    """                                                      
                            
    #This line saves the LDA model
    lda_model =gensim.models.ldamodel.LdaModel.load(pathtohere+'\\ldamodels\\ldaModel_10period_1gram')                            
   
    #print('---Printing 100 words per category (LDA)---')
    #pprint(lda_model.print_topics(num_words=100))
    
    print('Starting Dataframe for Dominant topics')
    #Dominant topic section
    sent_topics_df= pd.DataFrame()
    sent_topics_df=mlf.getDominantTopicDataFrame(lda_model,corpus,lsDocuments_NoSW,lsSubject)
    #mlf.generateFileSeparatedBySemicolon(sent_topics_df,'LDA_DominantTopic_Subject.txt')
    print(sent_topics_df.Text)

    #Generate graphs
    

if __name__=='__main__':
    main()    
