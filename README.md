## Document Clustering

A Use Case for testing document clustering based on text content using the Kleister-charity dataset.

Only a selection of the dataset is used to facilitate evaluation. The documents' texts are processed and lemmitized with NLTK. TF-IDF vectorisation is applied to test different clustering algoritms. And an attempt is made to evaluate the different clustering approaches.

### Table of contents

* [Intro](#intro)
* [Packages used](#packages-used)
* [Kleister-charity dataset](#kleister-charity-dataset)
* [Exploring data and preprocessing](#exploring-data-and-preprocessing)
* [Lemmitization](#lemmitization)
* [TF-IDF](#tf-idf)
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

I chose to work on the test set documents as the total file size is reduced and 'manual' evaluation of any clustering effort is still practical.

## Exploring data and preprocessing

See ```Exploring Kleister.ipynb```

## Lemmitization

## TF-IDF

## Clustering

## Evaluation

 
  
