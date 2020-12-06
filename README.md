## Document Clustering

A Use Case for testing document clustering based on text content using the Kleister-charity dataset.

Only a selection of the dataset is used to facilitate processing and evaluation. The documents' texts are processed and lemmitized with NLTK. TF-IDF vectorisation is applied to to create a cosine similarity matrix and to test different clustering algoritms. An attempt is made to evaluate the different clustering approaches.

### Table of contents

* [Intro](#intro)
* [Packages used](#packages-used)
* [Kleister-charity dataset](#kleister-charity-dataset)
* [Exploring data and preprocessing](#exploring-data-and-preprocessing)
* [Lemmitization](#lemmitization)
* [TF-IDF vectorisation](#tf-idf-vectorisation)
* [Document similarity](#document-similarity)
* [Clustering](#clustering)
* [Evaluation](#evaluation)

## Intro

The broader goal of this use case is to use document clustering to speed up the process of document annotation. In this way similar documents can be handled by same person which speeds up short term memory retention and improves consistency. 
- rather have 100% automation on 90% of the documents than 80% automation on 95% of the documents.
- we want to ignore outliers- documents that don't look like anything else- as they provide less value.
- the algorithm should be optimized for datasets 50-1000 documents, where specific “templates” can
occur between 2-500 times.
- detect clusters of documents that are very similar
- for large clusters, we want to limit the annotation to e.g. 20 docs until the model recognizes them correctly.
- flag duplicates or near duplicates (e.g. different scan of same document might have some other OCR artifacts, but be essentially the same) for
removal because they can make the results on the test set over-optimistic.
- for any given document, be able to query the top n most similar documents

The suggested approach is to use simple tf-idf based clustering. More advanced methods like document embeddings are possible too. Rather than optimizing the document embedding part it is advised to spend more time on tuning the clustering so that it automatically detects reasonable clusters. Good cut-offs are necessary to decide what is an outlier, which documents are similar, ...

## Packages used

- Numpy: a scientific computation package
- Pandas: a data analysis/manipulation tool using dataframes
- NLTK: a Natural Language Processing library 
- Sci-kit Learn: Machine Learning package that supports supervised/unsupervised learning and preprocessing

## Kleister-charity dataset

This dataset contains financial reports in the English language from a variety of British charities. The reports are processed with OCR and provided in TSV format. The original PDF files can be downloaded as well.

The dataset can be retrieved from : https://github.com/applicaai/kleister-charity
| train set | dev-0 set | test-A set |
|-------|-------| ------- |
|1729 items | 440 items | 609 items |

Clone the repo, install git-annex and download the actual reports as PDF files:
```
git clone https://github.com/applicaai/kleister-charity
sudo apt-get install git-annex
cd path/kleister-charity
```
```git-annex get --all --from pub-aws-s3.sh``` to download all documents (+12Gb)

```git-annex get-test-documents-from-s3.sh``` to download dev-0 and test-A test sets documents (+2Gb)

I chose to work on the test set documents, test-A, as the total file size is reduced and 'manual' evaluation of any clustering effort is still practical.

## Exploring data and preprocessing

See ```Exploring_Kleister.ipynb```

A subset of the Kleister-charity dataset, test-A  which consists of 609 documents, was selected to work on. The test-A tsv file contains the OCR'ed text for those 609 documents.

Of the 4 OCR approaches in the data, 'text_djvu', 'text_tesseract', 'text_textract' and 'text_best', the Tesseract processed text is used.

It was difficult to distinguish the best OCR rendering of the scanned PDF files but the Tesseract processing is among the best when looking at text quality. After inspecting the OCR'ed text preprocessing is applied by removing line breaks.

## Lemmitization

The document text further needs to be tokenized for vectorisation, here lemmitization was used specifically. Text is lower cased, punctuation removed and only tokens containing letters are kept. Lemmitization was done with NLTK's ```WordNetLemmatizer```. 
 ```
def tokenize_lem(text):
    lem = WordNetLemmatizer()
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.match('[a-zA-Z]', token) and not token in stop_words and len(token)>3:
            lemmatized_word = lem.lemmatize(token)
            filtered_tokens.append(lemmatized_word)
    return filtered_tokens
  ```
## TF-IDF vectorisation

See ```Kleister_TF-IDF.ipynb```

The 'text_tesseract' OCR column is vectorised using Sci-kit Learn's ```tfidf_vectorizer```.

TF-IDF, or term frequency - inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word.
TF-IDF = tf*idf

A maximum of 5000 features is set and the vectoriser will use the function described above for tokenization.
```
# Call tfidf_vectorizer with params -> point to 'tokenize_lem' func which lemmatizes and removes stop words
tfidf_vectorizer = TfidfVectorizer(max_df= 0.8, min_df= 0,max_features=5000, stop_words=None,
                                   use_idf=True, tokenizer=tokenize_lem, norm='l2')
X = tfidf_vectorizer.fit_transform(kl['text_tesseract']) # fit vectorizer and transform documents' text content to vectors
X.todense() # turn sparse matrix to dense
```
TF-IDF vectorisation returns a sparse matrix with 609 document rows and 5000 feature columns.

## Document similarity

See ```Kleister_TF-IDF.ipynb```

Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It is defined to equal the cosine of the angle between them, which is also the same as the inner product of the same vectors normalized to both have length 1. A cosine similarity of 0 means that the angle between two vectors equals 90 deg and the documents are not similar at all. A value of 1, for an angle of 0 deg, means that the documents are same. High cosine similarity values represent similar documents.

The cosine similarity matrix is calculated from the vectorisation matrix as follows:
```
cosim = (X * X.T).toarray()
```
Result is a matrix of 609x609 items of document-to-document cosine similarity value, with a value of 1 on the diagonal, representing the cosine similarity of a document with itself.

A histogram of the cosine similarity values:

<p align="center">
  <img src="https://github.com/kristof-becode/Document-Clustering/blob/master/img/histogram_cosim.png" width=75% >
</p>


By looking at the highest cosine similarity values in the dataset, besides the values on the matrix diagonal, we find 4 values >0.99999 representing the cosine similarity between two pairs of documents. 

When looking at the PDF files we can clearly state that these documents are 'identical':

* ```6b15787e2654b725f2bfc86da7dea511.pdf``` and ```e7d861735330f70a05d0aa51a5a4b096.pdf``` are two documents from The Housley Bequest Limited that are exactly identical but for the year of the report and the actual numbers.

* ```5d06055f6a4b58260fe2dcf6871db799.pdf``` and ```efac1f09a642532db1fb18b63e1f13b1.pdf``` are two identical fiancial reports for year 2015, only the ordering of two pages is slighty different, in one document they are found at the beginning of document, in the other at the end.¶

When looking a the lowest maximum cosine similarity for all documents we find a document with lowest max cosine sim < 0.1:

* ```b57e1ae7a9f286733362fa87fa704543.pdf``` appears to be a document that was scanned upside down and doesn't look like any other document.

These documents can be excluded from clustering.

## Clustering

 See ```Kleister_Cluster.ipynb```
 
 



## Evaluation

 
  
