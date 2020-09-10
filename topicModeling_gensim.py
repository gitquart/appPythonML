import MLfunctions as mlf
import gensim
from gensim import corpora
from pprint import pprint
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
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
                                           random_state=100,
                                           chunksize=100)

    pprint(lda_model.print_topics())
    


    



if __name__=='__main__':
    main()    
