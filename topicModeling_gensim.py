import MLfunctions as mlf
from gensim import corpora
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import gensim
import seaborn as sns
import matplotlib.colors as mcolors

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
    
    
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                id2word=id2word,
                                num_topics=5, 
                                random_state=100)
                                                          
                            
    #This line saves the LDA model
    #lda_model =gensim.models.ldamodel.LdaModel.load(pathtohere+'\\ldamodels\\ldaModel_10period_1gram')                            
   
    #print('---Printing 100 words per category (LDA)---')
    #print(lda_model.print_topics(num_words=100))
    
    print('Starting Dataframe for Dominant topics')
    #Dominant topic section
    sent_topics_df= pd.DataFrame()
    sent_topics_df=mlf.getDominantTopicDataFrame(lda_model,corpus,lsDocuments_NoSW,lsSubject)
    #mlf.generateFileSeparatedBySemicolon(sent_topics_df,'LDA_DominantTopic_Subject.txt')
    #Generate graphs
    """
    doc_lens = [len(d) for d in sent_topics_df.Text]

    # Plot
    plt.figure(figsize=(16,7), dpi=160)
    plt.hist(doc_lens, bins = 1000, color='navy')
    plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
    plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0,1000,9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
    plt.savefig(pathtohere+'\\wordsSpreadInAllDoc.png')
    plt.show()
    """
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    fig, axes = plt.subplots(2,2,figsize=(16,14), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):    
        df_dominant_topic_sub = sent_topics_df.loc[sent_topics_df.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        ax.hist(doc_lens, bins = 1000, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 1000), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i])
        ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0,1000,9))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
    plt.savefig(pathtohere+'\\wordsTopicSpreadInAllDoc.png')
    plt.show()

if __name__=='__main__':
    main()    
