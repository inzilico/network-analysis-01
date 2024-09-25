"""
Apply autoencoder to embeddings of PPI (protein-protein interaction) network 
to reduce the dimension from D to D', where D > D'.
Author: Gennady Khvorykh
Created: September 24, 2024
"""

import argparse
from helpers import load_embeddings, plot_loss
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from keras import layers, Input, Model, optimizers

# Initiate parser of command line arguments
parser = argparse.ArgumentParser(description="Apply AE to embeddings of PPI network")
parser.add_argument("-i", "--input", help="path/to/embeddings.txt", required=True)
parser.add_argument("-o", "--output", help="path/to/output_prefix", required=True)
parser.add_argument("--batch_size", help="batch_size", default=32, type=int)
parser.add_argument("--epochs", help="The number of epochs to train", default=100, type=int)
parser.add_argument("--units", help="The number of units of the code layer", default=32, type=int)

# Get command line arguments
args = parser.parse_args()

# Initiate variables
input_file = args.input
output = args.output
batch_size = args.batch_size
epochs = args.epochs
units = args.units
cpu = 20

# Load data
samples, data = load_embeddings(input_file)
# Get the number of features
ncol = data.shape[1]

# Instantiate a Keras tenzor
inputs = Input(shape=(ncol,), name="Input")
# Encoder layers
encoded1 = layers.Dense(units=32, activation='relu')(inputs)
# Decoder layers
decoded1 = layers.Dense(units=ncol, activation='sigmoid')(encoded1)

# Combine encoder and decoder
ae = Model(inputs=inputs, outputs=decoded1, name="AE")

# Compile the model
opt = optimizers.Adamax(learning_rate=0.001)
ae.compile(optimizer=opt, loss='mse', metrics=["accuracy"])
print(ae.summary())

# Train autoencoder
history = ae.fit(x=data, y=data, batch_size=batch_size, epochs=epochs, shuffle=False, 
       workers=cpu, use_multiprocessing=True, validation_split=0.2)

# Visualize and save learning history
plot_loss(history=history, output=output)

## Get data of reduced number of features
encoder = Model(inputs=inputs, outputs=encoded1, name="Encoder")
encoded = pd.DataFrame(encoder.predict(data), index=samples)

# Save output
encoded.to_csv(output + ".encoded.txt", sep=" ", header=False)
