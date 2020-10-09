import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download('wordnet')
from nltk.stem import PorterStemmer

# Display options dataframes
pd.set_option('display.width',400)
pd.set_option('display.max_columns', 40)
# Display options numpy arrays
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
# check for nans
kleister = pd.read_csv('/home/becode/AI/Data/Faktion/kleister-charity/test-A/in.tsv', sep='\t', names=['filename', 'keys', 'text_djvu', 'text_tesseract', 'text_textract', 'text_best'])
#print(kleister.head())
print(kleister.shape)
kleister = kleister.dropna()
print(kleister.shape)
kleister =kleister.drop(columns=['keys', 'text_djvu','text_textract','text_best'])
kleister['text_tesseract'] = kleister['text_tesseract'].astype(str)
kleister['text_tesseract'] = kleister['text_tesseract'].apply(lambda x: x.replace("\n",""))
kleister['text_tesseract'] = kleister['text_tesseract'].apply(lambda x: x.replace("\\n",""))
print(kleister.loc[:, ["text_tesseract"]].iloc[0].values)
def noNumbers(text):
    msg= text.split()
    for item in msg:
        if re.match("\d+", item):
            ret = msg.replace(item,"")
        else:
            ret = item
    return ret
#kleister['text_tesseract'] = kleister['text_tesseract'].apply(noNumbers(x))
#kleister['text_tesseract'] = kleister.apply(noNumbers(kleister['text_tesseract']),axis=1)#kleister['text_tesseract'].apply(noNumbers(x))
stop_words = set(stopwords.words('english'))

# Tokenize sentence and word
def tokenize_lem(text):
    lem = WordNetLemmatizer()
    stem = PorterStemmer()
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.match('[a-zA-Z]', token) and not token in stop_words and len(token)>3: # changed from if re.search('[a-zA-Z]', token)
            #filtered_tokens.append(token)
            lemmatized_word = lem.lemmatize(token)#, 'v')
            #stemmed = stem.stem(token)
            filtered_tokens.append(lemmatized_word)
            #filtered_tokens.append(stemmed)
    return filtered_tokens
kleister['text_tesseract'] = kleister['text_tesseract'].apply(tokenize_lem)
print(kleister.loc[:, ["text_tesseract"]].iloc[0].values)
print(len(kleister.loc[:, ["text_tesseract"]].iloc[0].values[0]))