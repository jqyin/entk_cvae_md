import numpy as np
from CVAE import CVAE
from sklearn.metrics import accuracy_score
import h5py 
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument("-i", default='cvae_input.h5', 
                    help="Input: contact map h5 file")
parser.add_argument("-w", default='cvae_weight.h5', 
                    help="Input: model weight h5 file")
parser.add_argument("-d", default=3, type=int, 
                    help="Number of dimensions in latent space") 
parser.add_argument("-o", default='cvae_output', 
                    help="Output: pred contact map and embedding npy file") 

args = parser.parse_args()
input_file=args.i
model_weight=args.w
hyper_dim=args.d
output_file=args.o

npred = 1000

cm_data = h5py.File(input_file,'r')['contact_maps'].value

cvae = CVAE(cm_data.shape[1:], hyper_dim)
cvae.model.load_weights(model_weight)

cm_pred = np.array([cvae.decode(np.expand_dims(sample,axis=0)) for sample in cm_data[:npred]])
cm_emb = np.array([cvae.return_embeddings(np.expand_dims(sample,axis=0)) for sample in cm_data])

np.savez(output_file, pred=cm_pred, emb=cm_emb)
cm_pred = cm_pred.astype(cm_data.dtype)

acc = accuracy_score(cm_data[:npred].flatten(),cm_pred[:npred].flatten())
print("Accuracy: ", acc)

