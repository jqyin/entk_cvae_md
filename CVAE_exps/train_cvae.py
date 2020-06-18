import os, sys, errno
import argparse 
from cvae.CVAE import run_cvae  
import numpy as np 


parser = argparse.ArgumentParser()
parser.add_argument("--train_file", default='cvae_input.h5', help="Input: contact map h5 file")
parser.add_argument("--val_file", default='cvae_input.h5', help="Input: contact map h5 file")
parser.add_argument("-d", "--dim", default=3, help="Number of dimensions in latent space")
parser.add_argument("-batch_size", default=32, type=int, help="batch size")
parser.add_argument("-epochs", default=100, type=int, help="number of epochs")

args = parser.parse_args()

train_file = args.train_file
val_file = args.val_file
hyper_dim = int(args.dim) 
batch_size = args.batch_size
epochs = args.epochs

if not os.path.exists(train_file):
    raise IOError('Input file doesn\'t exist...')


if __name__ == '__main__': 
    cvae = run_cvae(train_file, val_file, hyper_dim=hyper_dim, 
                    batch_size=batch_size, epochs=epochs)

