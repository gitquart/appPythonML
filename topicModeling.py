import os
import nltk
import pandas as pd
#sent or word tokenize: Get the information into sentences or words
import matplotlib.pyplot as plt
import MLfunctions as mlf
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
import random
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

"""
Topic Modeling
-Latent Dirichlet Allocation (LDA)
-Non-Negative Matrix Factorization (NMF)

-It can be used with TD IDF or Normal Vector
-Test 5 groups for each algorithm
"""

pathtohere=os.getcwd()


def main():
    print('1.LDA, 2.NMF,3.LSA')
    op=input()
    op=int(op)
    if op==1:
        lsRes=[]
        lsRes=mlf.getCountVectorizer()
        vector=lsRes[0]
        lsDocs=lsRes[1]
        term_freq_matrix=vector.fit_transform(lsDocs)
        print('LDA...')
        LDA = LatentDirichletAllocation(n_components=5)
        LDA.fit(term_freq_matrix)
        for i,topic in enumerate(LDA.components_):
            print(f'Top 100 words for topic #{i}:')
            print([vector.get_feature_names()[i] for i in topic.argsort()[-100:]])
            print('\n')
        

    if op==2:     
        lsRes=[]
        lsRes=mlf.get_TFIDF()
        tfidf=lsRes[0]
        lsDocs=lsRes[1]
        tfidf_matrix=tfidf.fit_transform(lsDocs)
        print('TF IDF shape:',tfidf_matrix.shape)
        print('NMF...')
        nmf = NMF(n_components=5,init='random')
        nmf.fit(tfidf_matrix)
        for i,topic in enumerate(nmf.components_):
            print(f'Top 100 words for topic #{i}:')
            print([tfidf.get_feature_names()[i] for i in topic.argsort()[-100:]])
            print('\n')  

    if op==3:
        lsRes=[]
        lsRes=mlf.get_TFIDF()
        tfidf=lsRes[0]
        lsDocs=lsRes[1]
        tfidf_matrix=tfidf.fit_transform(lsDocs)
        print('TF IDF shape:',tfidf_matrix.shape)
        print('LSA...')







if __name__=='__main__':
    main()