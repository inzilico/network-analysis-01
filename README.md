# Gene Network Analysis 

## Description

The repo contains the scripts to analyse the networks of genes (protein-protein interactions, PPI) for the following tasks: 

* Network Representation Learning (NRL)
* Vizualization 
* Prediction of new genes

* **helpers.py** contains custom functions applied in the scripts from this repo

## Network Representation Learning

* **deep-walk-01.py** creates embeddings of the graph with DeepWalk implemented in the library [karateclub](https://github.com/benedekrozemberczki/karateclub). 

## Prediction of new genes

* **add-outcome-01.py** add the column with gene labels (1 for genes with known label and 0 for unlabeled data)



