"""
DeepWalk algorithm based on gensim.Word2Vec
Author: Gennady Khvorykh
Created: July 11, 2024
"""

from tqdm import tqdm
import networkx as nx
from gensim.models import Word2Vec
import time
from helpers import get_random_walk, show_time_elepsed 
import os, sys
import argparse
from multiprocessing.pool import Pool

# === Functions ===
def make_random_walks(node, G, walk_numbers, walk_length):
   random_walks = []
   for i in range(walk_numbers):
      random_walks.append(get_random_walk(G, node, walk_length))      
   return(random_walks)

# === End of functions ===

# Initiate argument parser
parser = argparse.ArgumentParser(description="Make embeddings of the graph with DeepWalk \
                                 implemented as custom random walks and model fitting with gensim.Word2Vec.")
parser.add_argument("-i", "--input", help="path/to/filename with agjacency list", required=True)
parser.add_argument("-o", "--output", help="path/to/filename to save embeddings", required=True)
parser.add_argument("-d", "--dimensions", help="The number of dimensions (default=64)", 
                    default=64, type=int)
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

# Create graph by loading adjacency list from file
G = nx.read_adjlist(path=input_file)
print("Nodes: ", len(G.nodes))
print("Edges: ", len(G.edges))

# Get nodes
nodes = list(G.nodes)

# Create iterable arguments 
items = [(node, G, walk_number, walk_length) for node in nodes]

# Get random walks for each node
print("Getting random walks...")
with Pool(cpu) as pool:
   results = pool.starmap(make_random_walks, items)

# Flatten
random_walks = [x for xs in results for x in xs]

# Show the number of random walks created
print("No of random walks:", len(random_walks))

# Initiate the model
model = Word2Vec(vector_size=dimensions, alpha=0.05, min_count=1, seed=42, workers=cpu, hs=1, epochs=1)

# Build vocabularly
model.build_vocab(random_walks, progress_per=2)
print("Corpus count:", model.corpus_count)

# Train the model
print("Fitting Word2Vec model...")
model.train(random_walks, total_examples=model.corpus_count, epochs=20, report_delay=1)

# Save embeddings into file
with open(output_file, "w") as f:
   for i, k in enumerate(model.wv.index_to_key):
      vec = " ".join([str(x) for x in model.wv[k]])
      print(k, vec, file=f)

# Show time elepsed
show_time_elepsed(ts=ts)