"""
Add column with outcomes to embeddings.
Shuffle labeles of outcome variable n times if required.
Author: Gennady Khvorykh
Created: September 29, 2024
"""

import pandas as pd
from helpers import check_input, check_output_folder
from random import shuffle
import argparse

def parser_arguments():
    parser = argparse.ArgumentParser(prog="add_outcome-01.py",
                                     description="Add column with outcome variable to embeddings and shuffle labels if required")
    parser.add_argument("-e", "--embeddings", 
                        help="path/to/filename.txt with embeddings. The first column has the node ids", 
                        required=True)
    parser.add_argument("-l", "--labels", 
                        help="path/to/filename.txt containing table whith columns 'Gene' and 'Gene_id'", 
                        required=True)
    parser.add_argument("-o", "--output_prefix", help="path/to/prefix to save output file(s)", required=True)
    parser.add_argument("-r", "--random", 
                        help="A flag to shuffle the outcome labeles if chosen. Default: not shuffle",
                        action="store_true")
    parser.add_argument("-n", "--n_max", help="Maximum number of permuted outcomes to be genrated. Default: 100",
                        default=100, type=int)
    
    return(parser.parse_args())

def save_output(dt, outcome, output_file):
    # Add column with outcome 
    dt.insert(loc=1, column="outcome", value=outcome)
    # Save into file
    dt.to_csv(output_file, sep=" ", header=False, index=False)

def main():
    # Get command line arguments
    args = parser_arguments()
    input_file1 = args.embeddings 
    input_file2 = args.labels 
    output_prefix = args.output_prefix 
    random = args.random
    n_max = args.n_max

    # Check input
    check_input([input_file1, input_file2])
    check_output_folder(output_prefix)
        
    # Load data
    d1 = pd.read_csv(input_file1, header=None, sep=" ")
    print(f'{input_file1}:', d1.shape)
    d2 = pd.read_csv(input_file2, sep = "\t", usecols=["Gene", "Gene_id"])
    print(f'{input_file2}:', d2.shape)
    
    # Get common nodes for columns with node ids from two tables
    common = list(set(d1[0]) & set(d2["Gene_id"]))
    
    # Show the number of common nodes
    print("No of common nodes:", len(common))
    
    # Create outcome variable
    outcome = [1 if x in common else 0 for x in d1[0]]
    
    # Shuffle outcomes id require
    if random:
        for i in range(n_max):        
            shuffle(outcome)
            save_output(d1.copy(), outcome, f"{output_prefix}-{i}.txt")
        print("Output files with shuffled outcomes:", n_max)    
    else:
        output_file = output_prefix + ".txt"
        save_output(d1, outcome, output_file)
        print("Output file saved:", output_file)    
        
    
if __name__ == "__main__":
    main()