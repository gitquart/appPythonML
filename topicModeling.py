import os
import nltk
import pandas as pd
#sent or word tokenize: Get the information into sentences or words
import matplotlib.pyplot as plt
import MLfunctions as mlf
from sklearn.decomposition import NMF
import random


pathtohere=os.getcwd()


def main():
    lsRes=[]
    lsRes=mlf.get_TFIDF()
    tfidf=lsRes[0]
    lsDocs=lsRes[1]
    tfidf_matrix=tfidf.fit_transform(lsDocs)
    nmf = NMF(n_components=5, random_state=42)
    nmf.fit(tfidf_matrix)
    for i,topic in enumerate(nmf.components_):
        print(f'Top 10 words for topic #{i}:')
        print([tfidf.get_feature_names()[i] for i in topic.argsort()[-10:]])
        print('\n')  






if __name__=='__main__':
    main()