import numpy as np
import h5py 
import keras 

class CVAEGenerator(keras.utils.Sequence):

    def __init__(self,cm_file,hvd_size=1, batch_size=32,shuffle=True):
        cm_h5 = h5py.File(cm_file, 'r', libver='latest', swmr=True)
        self.cm_data = cm_h5[u'contact_maps']
        self.batch_size = batch_size
        self.hvd_size = hvd_size
        self.shuffle = shuffle
        self.input_shape = self.cm_data.shape 
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.cm_data)/(self.hvd_size*self.batch_size)))

    def __getitem__(self, idx):
        idxes = self.idxes[idx*self.batch_size:(idx+1)*self.batch_size]
        data_batch = np.array([ self.cm_data[e] for e in idxes ]) # self.cm_data[list(idxes)] 
        return data_batch, data_batch

    def get_shape(self):
        return self.input_shape 

    def on_epoch_end(self):
        self.idxes = np.arange(len(self.cm_data))
        if self.shuffle == True:
            np.random.shuffle(self.idxes)

