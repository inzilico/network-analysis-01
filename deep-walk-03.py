"""
Make embeddings from adjacency matrix of the graph with DeepWalk implemented in karateclub.
https://github.com/benedekrozemberczki/karateclub
Author: Gennady Khvorykh
Created: October 14, 2024
"""

import networkx as nx
import sys, os
from karateclub import DeepWalk
import pandas as pd
import helpers
import time
import argparse
import h5py

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

# Import adjacency matrix
with h5py.File(input_file, "r") as h5:
    adj_matrix = helpers.attach_matrix(h5)[:]

# Create graph from agjacency list 
print("Creating graph...")
G = nx.from_numpy_array(adj_matrix)
print("Nodes: ", len(G.nodes))
print("Edges: ", len(G.edges))

# Crete estimator  
model = DeepWalk(walk_number=walk_number, 
                 walk_length=walk_length, 
                 dimensions=dimensions, 
                 workers=cpu) 

# Train model
print("Training model...")
model.fit(G) 

# Get embeddings
embedding = model.get_embedding() 

# Show embedding shape 
print("Embedding shape:", embedding.shape)

# Convert into data frame 
df = pd.DataFrame(embedding, index=list(G.nodes))

# Save embeddings into file
df.to_csv(output_file, sep=" ", header=False, index=True)

# Show time elepsed
helpers.show_time_elepsed(ts)