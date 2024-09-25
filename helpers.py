"""
Custom functions to assisit data processing. 
Author: Gennady Khvorykh 
"""

import networkx as nx
import time
import random
import os, sys
import pandas as pd
from sklearn.preprocessing import minmax_scale
from math import log10, ceil
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import roc_auc_score, f1_score, classification_report 
from sklearn.svm import SVC


def relabel_nodes(G: nx.graph) -> nx.graph:
    """
    Relabel nodes by consecutive integers if dictionary not provided or by 
    values in dictionary.
    """
    # Get node labels
    nodes = list(G.nodes)
    # Initiate dictionary to keep old (keys) and new (values) node labels
    mapping = {}
    # Loop over all nodes
    for i in range(len(nodes)):
        # Fill dictionary 
        mapping[nodes[i]] = i
        
    # Relabel nodes in place
    nx.relabel_nodes(G=G, mapping=mapping, copy=False)
    return(G, mapping)
    
def show_time_elepsed(ts):
    # Show time elepsed
    dur = time.time() - ts
    s = time.strftime("%H:%M:%S", time.gmtime(dur))
    print("\nTime elapsed:", s)
    
def get_random_walk(g: nx.graph, node: str, walk_length: int) -> list:
    # Initiate variable
    random_walk = [node]
    # Iterate to build random walk of given path length
    for _ in range(walk_length - 1):
        neighbors = list(g.neighbors(node))
        neighbors = list(set(neighbors) - set(random_walk))
        if len(neighbors) == 0: break
        random_node = random.choice(neighbors)
        random_walk.append(random_node)
        node = random_node
    return random_walk

def check_input(files: list):
    for file in files:
        if not os.path.isfile(file):
            print(file, "doesn't exist")
            sys.exit(1)

def load_embeddings(x: str) -> tuple:
    """
    Load and scale the table of features, say, embeddings.
    Args: 
        x: path/to/filename.txt with objects in rows and features in columns.
        The first column is the object ids. 
    Return: 
        A tuple with object ids and ndarray with scaled features.
    """
    
    # Check input
    check_input([x])

    # Load input data
    d1 = pd.read_csv(x, sep=" ", header=None)
    print("\nInput:", x)
    print(f"Shape: {d1.shape[0]} x {d1.shape[1]}\n")

    # Subset samples and data (embeddings)
    samples = d1[0]
    d2 = d1.iloc[:,1:]

    # Scale data
    d3 = minmax_scale(X=d2)
    return samples, d3

def load_data(x: str):
    # Check input
    check_input([x])
    # Load data
    data = pd.read_csv(x, sep=" ", header=None)

    # Subset features, labels, and samples
    X = data.iloc[:, 2:]
    y = data[1]
    samples = data[0]

    return(X, y, samples)
 
def epd(df: pd.DataFrame):
     """
     Run edge pattern detection 
     """
     # Convert input to Numpy array
     x = df.to_numpy(copy=True)
     print("Data:", x.shape)
     
     # Estimate k
     N = x.shape[0]
     k = ceil(5 * log10(N))
     
     # Find k nearest neighbors
     knn = NearestNeighbors(n_neighbors=k)
     knn.fit(x)
     nn = knn.kneighbors(n_neighbors=k, return_distance=False)
         
     # Initiate variables
     T = 0.1
     x_edge = []
     
     for i, vec in enumerate(x):
        # Calculate k-nn direction vector
        v_i = []
        for j in nn[i]:
            v_ij = vec-x[j]
            v_i.append(v_ij/np.linalg.norm(v_ij))
        # Approximate normal vector
        n_i = sum(v_i)
        # Calculate theta
        theta = [np.dot(np.transpose(v), n_i) for v in v_i] 
        # Calculate l
        l = 1/k*sum([t >= 0 for t in theta])
        if l >= 1 - T:
            x_edge.append(vec)
        
     return x_edge

def partition (lst: list, n: int) -> list:
    """
    Shuffle and partition the list 
    """
    # Shuffle the list
    random.shuffle(x=lst)
    # Get devision
    division = len(lst) / n
    chunks =  [lst[round(division * i):round(division * (i + 1))] for i in range(n)]
    return(chunks)

def classify(X, y, clf="SVM"):
    """
    Apply binary clussificaton with SVM.
    """
    
    # Initiate variables
    cpu = 10
    
    # Split data into train and test ones
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.33)
    
    # Define estimator and parameter ranges for grid search 
    if clf == "SVM":
        estimator = SVC(probability=True)
        parameters = {"C": [0.1, 1, 10, 100, 1000],
                  "gamma": [1, 0.1, 0.01, 0.001, 0.0001]} 
    
    # Initiate the model 
    model = GridSearchCV(estimator=estimator, param_grid=parameters, n_jobs=cpu) 
  
    # Fitting the model for grid search 
    model.fit(X_train, y_train) 
    
    # Eveluate the model
    y_pred = model.predict(X_test)
    cr = classification_report(y_test, y_pred, zero_division=0, target_names=['class 0', 'class 1'])
    score = model.score(X_test, y_test)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return(model, cr, score, auc, f1)

def read_resources(x: str):
    """
    Load csv file and return a dictionary
    """
    check_input([x])
    df = pd.read_csv(x, header=None, sep=",", comment="#")
    return(dict(df[[0, 1]].values))

def plot_history(history, output: str):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE")
    ax1.plot(history.history["loss"], label="Training")
    ax1.plot(history.history["val_loss"], label="Validation")
    ax1.legend(loc='center right')
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(history.history["accuracy"], color="tab:green", label="Accuracy")
    fig.tight_layout()
    plt.savefig(output + ".history.png")
    
def plot_loss(history, output: str):
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(output + ".loss.png")