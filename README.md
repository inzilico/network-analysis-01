# Network Analysis 

## Description

The repo provides custom scripts to analyse the networks for the following tasks: 

* Network Representation Learning (NRL)
* Vizualization of the networks
* Novelty detection 

The scripts were originally created and applied to analyze the networks of genes. 

## General scripts and files

* **helpers.py** contains custom functions applied in the scripts from this repo

* **res.csv** contains comma seperated lists each having program name, path to the executable file and optionally the progrma version. 

## Network Representation Learning

* **deep-walk-01.py** creates embeddings of the graph with DeepWalk implemented in the library [karateclub](https://github.com/benedekrozemberczki/karateclub). 

* **deep-walk-02.py** creates embeddings of the graph with custom random walks and Word2vec from [Gensim](https://radimrehurek.com/gensim/) library.  

## Autoencoders

* **ae-01.py** applies an autoencoder with one hidden layer to embeddings.

## Novelty detection

The task is to predict the label of unlabeled nodes of the graph given a set of labeled nodes. 

* **add-outcome-01.py** adds the column with labels (1 for known label and 0 for unlabeled data) to table with embeddings.

* **run-binary-classification-01.py** samples randomly the negative class and applies binary classification (SVM, RF, AdaBoost, MLPC) to predict the label of unlabeled nodes. 


