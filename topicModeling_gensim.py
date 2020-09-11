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
    #Iterate each row of DataFrame and add to it a semicolon to load it to excel
    fileContent=''
    fileContent=sent_topics_df.columns[0]+';'+sent_topics_df.columns[1]+';'+sent_topics_df.columns[2]+';'+sent_topics_df.columns[3]+';'+sent_topics_df.columns[4]+';\n'
    mlf.appendInfoToFile(pathtohere,'\\LDA_DominantTopic_Subject.txt',fileContent)
    for index, row in sent_topics_df.iterrows():
        fileContent=str(row['Topic_No'])+' ;'+str(row['Dominant_Topic'])+' ;'+str(row['Keywords'])+' ;'+str(row['Text_Without_stopwords'])+' ;'+str(row['Subject'])+' ;\n'
        mlf.appendInfoToFile(pathtohere,'\\LDA_DominantTopic_Subject.txt',fileContent)

if __name__=='__main__':
    main()    
