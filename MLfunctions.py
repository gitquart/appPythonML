from io import StringIO

def convertListToString(lst):
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