# Gene Network Analysis 

## Description

The repo provides the scripts to analyse the networks of genes for the following tasks: 

* Network Representation Learning (NRL)
* Vizualization 
* Prediction of new genes

The networks of genes represent the protein-protein interactions (PPI).

## General scripts

* **helpers.py** contains custom functions applied in the scripts from this repo

## Network Representation Learning

* **deep-walk-01.py** creates embeddings of the graph with DeepWalk implemented in the library [karateclub](https://github.com/benedekrozemberczki/karateclub). 

* **deep-walk-02.py** creates embeddings of the graph with custom random walks and Word2vec from [Gensim](https://radimrehurek.com/gensim/) library.  

## Autoencoders

* **ae-01.py** applis an autoencoder with one hidden layer to embeddings.

## Prediction of new genes

* **add-outcome-01.py** adds the column with gene labels (1 for genes with known label and 0 for unlabeled data)
* **run-binary-classification-01.py** samples negative class and apply binary classification (SVM, RF) to predict new genes. 


