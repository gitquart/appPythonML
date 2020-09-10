import MLfunctions as mlf
import gensim
from gensim import corpora
from pprint import pprint
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import pandas as pd
sw=stopwords.words('spanish')


def main():
    print('LDA model with gensim, 1-gram')
    lsDocuments=[]
    #Get the the information into a list of documents
    lsDocuments=mlf.getRawTextToList()
    lsDocuments_NoSW = [[word for word in simple_preprocess(str(doc)) if word not in sw] for doc in lsDocuments]
    # Create Dictionary
    id2word = corpora.Dictionary(lsDocuments_NoSW)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in lsDocuments_NoSW]
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                id2word=id2word,
                                num_topics=5, 
                                random_state=100)
   
    #print('---Printing 100 words per category (LDA)---')
    #pprint(lda_model.print_topics(num_words=100))
    
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
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(lsDocuments_NoSW)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df.reset_index()
    sent_topics_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    sent_topics_df.head(10)

if __name__=='__main__':
    main()    
