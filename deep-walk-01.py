"""
Make embeddings of the graph with DeepWalk implemented in karateclub.
https://github.com/benedekrozemberczki/karateclub
The nodes are reindexed. 
Author: Gennady Khvorykh
Created: July 10, 2024
"""

import networkx as nx
import sys, os
from karateclub import DeepWalk
import numpy as np
import pandas as pd
import helpers
import time
import argparse


# Initiate argument parser
parser = argparse.ArgumentParser(description="Make embeddings of the graph with DeepWalk \
                                 implemented in karateclub.")
parser.add_argument("-i", "--input", help="path/to/filename with agjacency list", required=True)
parser.add_argument("-o", "--output", help="path/to/filename to save embeddings", required=True)
parser.add_argument("-d", "--dimensions", help="The number of dimensions (default=128)", 
                    default=128, type=int)
parser.add_argument("-c", "--cpu", help="The number of CPU (default=10)", 
                    default=10, type=int)
parser.add_argument("-l", "--walk_length", help="The number of nodes in walk path (default=80)", 
                    default=80, type=int)
parser.add_argument("-n", "--walk_number", help="The number of random walks for each node (default=10)", 
                    default=10, type=int)

# Get command line arguments
args = parser.parse_args()

# Initiate variables
input_file = args.input
output_file = args.output
cpu = args.cpu
dimensions = args.dimensions
walk_length = args.walk_length
walk_number = args.walk_number
ts = time.time()

# Check input
if not os.path.isfile(input_file):
    print(input_file, "doesn't exist!")
    sys.exit(1)

# Show input
print("Input:", input_file)
print("Output:", output_file)
print("Dimensions:", dimensions)
print("CPU:", cpu)

# Create graph from agjacency list 
G1 = nx.read_adjlist(path=input_file)
print("Nodes: ", len(G1.nodes))
print("Edges: ", len(G1.edges))

# Relabel nodes by consecutive integers
G2, mapping = helpers.relabel_nodes(G1)

# Crete estimator  
model = DeepWalk(walk_number=walk_number, walk_length=walk_length, dimensions=dimensions, workers=cpu) 

# Train model
print("Training model...")
model.fit(G2) 

# Get embeddings
embedding = model.get_embedding() 

# Show embedding shape 
print("Embedding shape:", embedding.shape)

# Convert into data frame 
df = pd.DataFrame(embedding, index=mapping.keys())

# Save embeddings into file
df.to_csv(output_file, sep=" ", header=False, index=True)

# Show time elepsed
helpers.show_time_elepsed(ts)