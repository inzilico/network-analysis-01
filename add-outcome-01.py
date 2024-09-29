"""
Add column with outcomes to embeddings.
Author: Gennady Khvorykh
Created: July 15, 2024
"""

import pandas as pd
from helpers import check_input, check_output_folder
from collections import Counter
import sys

def main():
    # Get command line arguments
    input_file1 = sys.argv[1] # path/to/embeddings.txt
    input_file2 = sys.argv[2] # path/to/file.txt with the list of nodes (genes) with known labels (association).
    output_file = sys.argv[3] # path/to/file.txt to save output

    # Check input
    check_input([input_file1, input_file2])
    check_output_folder(output_file)
        
    # Load data
    d1 = pd.read_csv(input_file1, header=None, sep=" ")
    print(f'{input_file1}:', d1.shape)
    d2 = pd.read_csv(input_file2, sep = "\t", usecols=["Gene", "Gene_id"])
    print(f'{input_file2}:', d2.shape)
    
    # Left join
    d3 = pd.merge(left=d1, right=d2, how="left", left_on=0, right_on="Gene_id", sort=False, indicator="Source")

    # Create outcome variable
    outcome = [1 if x == "both" else 0 for x in d3["Source"]]
    print(Counter(outcome))

    # Add column with outcome 
    d1.insert(loc=1, column="outcome", value=outcome)

    # Save into file
    d1.to_csv(output_file, sep=" ", header=False, index=False)

    
if __name__ == "__main__":
    main()