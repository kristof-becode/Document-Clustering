Use Case for text clustering based on the kleister-charity dataset.

The dataset can be retrieved from : https://github.com/applicaai/kleister-charity
This set contains mainly financial reports from British charities.

The objective was to use TF-IDF to vectorise the document's text and to further cluster the documents with a suitable clustering algorithm.
Creating a pipeline with the necessary text preprocessing, tf-idf vectorisation and clustering on this specific datatset could prove to be useful for real-life document clustering to reduce annotation time.

This project is still a work in progress:
- more clustering algorithms need to be tested, possibly hierarchical clustering
- the preprocessing and text tokenization/lemmitization can be still be optimised
- the number of clusters still needs to be optilised further
- the pipeline still needs to be created for the scaling to other datsets or real-world use
- the evaluation of the clustering lacks in clarity

 
  
