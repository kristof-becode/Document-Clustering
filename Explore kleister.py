import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
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
#kleister['text_tesseract'] = kleister['text_tesseract'].apply(tokenize_lem)


#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df= 0.95, min_df= 0,max_features=5000, stop_words=None,use_idf=True, tokenizer=tokenize_lem)#, ngram_range=(1,3))
X = kleister['text_tesseract'] #max_df= 0.8, min_df= 0.2, max_df= 0.95, min_df= 0.05,
tfidf_matrix = tfidf_vectorizer.fit_transform(X) #fit the vectorize to synopses

print(tfidf_matrix.toarray())
df1 = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
print(df1)

print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
#print(terms)
print(tfidf_vectorizer.vocabulary_)
print(len(tfidf_vectorizer.vocabulary_))
X_reduced = tfidf_matrix
#print(kleister.head())
#print(kleister.loc[:, ["text_tesseract"]].iloc[0].values)
print("\n)")
#svd = TruncatedSVD(n_components=2, n_iter=10, random_state=42)
#X_reduced = svd.fit_transform(tfidf_matrix)
"""
# Elbow curve : inertia vs k
n_clusters=300
cost=[]
for i in range(1,n_clusters):
    kmeans= KMeans(i)
    kmeans.fit(tfidf_matrix)
    cost.append(kmeans.inertia_)
plt.plot(range(1,20),cost)
plt.xticks(range(1,20,2))
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
"""
#pca = PCA(n_components=0.95)
#X_reduced = pca.fit_transform(tfidf_matrix)

n_clusters=300
sil_scores=[]
for i in range(2,n_clusters):  # n_clusters can not be 1, took me a really long time to change the range from range(1, n) to (2,n)
    kmeans = KMeans(i)
    labels = kmeans.fit_predict(X_reduced) # or kmeans.labels_ is the same
    sil_scores.append(silhouette_score(X_reduced,labels))
plt.plot(range(2,300),sil_scores)
plt.title('The Silhouette curve')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')
plt.show()
"""
# Elbow curve : inertia vs k
n_clusters=20
cost=[]
for i in range(1,n_clusters):
    kmeans= KMeans(i)
    kmeans.fit(tfidf_matrix)
    cost.append(kmeans.inertia_)
plt.plot(range(1,20),cost)
plt.xticks(range(1,20,2))
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()



kmeans = KMeans(n_clusters=100, init='k-means++',n_init=10, max_iter=300, random_state=123 ) #287
kmeans.fit(X_reduced)
y=kmeans.predict(X_reduced)
print(y)
print(f"labels: {kmeans.labels_}") # y and kmeans.labels_ is same
clusters = kmeans.labels_.tolist()
print(clusters)
print(f"silhouette score = {silhouette_score(tfidf_matrix ,kmeans.labels_)}")
kleister['cluster'] = clusters
print(kleister['cluster'].value_counts())
kl = kleister.groupby('cluster')
print(kl)




dbscan = DBSCAN(eps=3, min_samples=2)
dbscan.fit_predict(tfidf_matrix)
print(dbscan.labels_)
#print(f"silhouette score = {silhouette_score(tfidf_matrix ,dbscan.labels_)}")

EXAMPLE_TEXT = str(kleister.loc[:, ["text_tesseract"]].iloc[0].values)
stop_words = set(stopwords.words('english'))
#kleister['text_tesseract']= kleister['text_tesseract'].apply(lambda x : x.lower())
feed = str(kleister['text_tesseract'].values)

sentoken = sent_tokenize(feed)
print(sentoken)
tokenizer = RegexpTokenizer(r'\w+')
regsentoken = tokenizer.tokenize(str(sentoken))
print(regsentoken)
print(len(regsentoken))
doc = []
for item in regsentoken:
    token = word_tokenize(item)
    if not token in stop_words:
        doc.append(token)
        print(token)
print(doc)

tfidf = TfidfVectorizer(input=doc, encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=False, preprocessor=None, tokenizer=None, analyzer='word', stop_words='english')
#tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
X = tfidf.fit_transform(doc)
print(vectorizer.get_feature_names())
print(X.shape)

kmeans = KMeans(n_clusters=3, init='k-means++',n_init=10, max_iter=300, random_state=123 )
kmeans.fit_predict(X)
print(f"silhouette score = {silhouette_score(X,kmeans.labels_)}")

"""
