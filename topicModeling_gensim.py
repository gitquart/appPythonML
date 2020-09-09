import MLfunctions as mlf
import gensim
from gensim import corpora


def main():
    print('LDA model with gensim')

    lsDocuments=[]
    lsDocuments=mlf.getCorpusList()
    texts = [[text for text in doc.split()] for doc in lsDocuments]
    dictionary = corpora.Dictionary(texts)
    print('...')



if __name__=='__main__':
    main()    
