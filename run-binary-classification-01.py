"""
Apply binary classification for novelty detection.
Author: Gennady Khvorykh
Created: September 14, 2024
"""


from helpers import check_input, partition, classify
import argparse
import pandas as pd
from collections import Counter
import numpy as np
import time

# Initiate argument parser
parser = argparse.ArgumentParser(
    prog="run-binary-classification-01.py", 
    description="Apply binary classification for novelty detection."
)
parser.add_argument("-i", "--input", help="path/to/filename.txt with input dataset", required=True)
parser.add_argument("-o", "--output", help="path/to/prefix to save output files", required=True)
parser.add_argument("-c", "--clf", help="A string denoting the clussifier (SVM, RF) (default: SVM)", 
                    default="SVM")

# Get command line arguments
args = parser.parse_args()

# Initiate variables
input_file = args.input
output = args.output
clf = args.clf
cpu = 10
ts = time.time()

# Check input
check_input([input_file])

# Open file for logging
fh1 = open(output + ".log", "w")
print("Input:", input_file, file=fh1)
print("Output:", output, file=fh1)
print("Clussifier:", clf, file=fh1)

# Load input dataset
df1 = pd.read_csv(input_file, sep=" ", header=None)
print("Input data:", df1.shape, file=fh1)

# Subset labels
labels = df1[1].to_list()
size_unlabeled = Counter(labels)[0] 
size_pos = Counter(labels)[1]
print("Positive class:", size_pos, file=fh1)
print("Unlabeled data:", size_unlabeled, file=fh1)

# Subset indexes of unlabeled data
ind = [True if label == 0 else False for label in labels]
unlabeled = df1.index[ind].to_list()

# Split list into chunks with random content of elements
n = round(size_unlabeled / size_pos)
unlabeled_chunks = partition(unlabeled, n) 

# Open file for model QC
fh2 = open(output + ".qc", "w")
print("chunk", "score", "auc", "f1", file=fh2)

# Apply binary classification to each unlabeled chunk
for i, chunk in enumerate(unlabeled_chunks):
    
    # Show which chunk being processed
    print("Processing chunk", i, "...")
    
    # Subset labeled data (class 1)
    cl1 = df1[df1[1] == 1]
    # Subset data by indexes in the chunk (class 0) 
    cl0 = df1.iloc[chunk]
    # Create data set for fitting the model
    df2 = pd.concat([cl0, cl1], axis = 0)
    # Subset matrix with features
    X = df2.iloc[:,2:]
    # Define outcome variable
    y = df2[1]
    # Apply binary classification
    model, cr, score, auc, f1 = classify(X=X, y=y, clf=clf, cpu=cpu)
    # Save the results of model fit to log file
    print(f"Chunk {i}: Class 0={cl0.shape[0]}, AUC={round(auc, 4)}, ACC={round(score, 4)}", file=fh1)
    print("Best parameters:", model.best_params_, file=fh1)
    print(cr, file=fh1)
    # Save the model QC 
    print(i, score, auc, f1, file=fh2)
    # Subset unlabeled data left for prediction
    set1 = set(unlabeled)
    set2 = set(chunk)
    diff = list(set1 - set2)
    df3 = df1.iloc[diff]
    # Predict
    arr = model.predict_proba(df3.iloc[:, 2:])
    prob = [x[1] for x in arr]
    # Merge the results from each chunk
    df4 = pd.DataFrame({"gene" : df3[0].to_list(), f"prob{i}" : prob})
    if i == 0:
        out = df4
    else:
        out = out.merge(df4, how="left", on="gene", sort=False)

# Close file with model QC
fh2.close()

# Add mean and SD of probabilities for each row
out["mean"] = out.iloc[:, 1:].apply(np.nanmean, axis=1)
out["std"] = out.iloc[:, 1:].apply(np.nanstd, axis=1)

# Save results obtained
out.to_csv(output + ".prob.txt", sep=" ", index=False, header=True)

# Show number of records saved
print("Records saved:", out.shape[0], file=fh1)

# Show time elepsed
dur = time.time() - ts
s = time.strftime("%H:%M:%S", time.gmtime(dur))
print("\nTime elapsed:", s, file=fh1)

# Close the log file
fh1.close()
