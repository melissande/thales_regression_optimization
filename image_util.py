from __future__ import print_function, division, absolute_import, unicode_literals


import numpy as np

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.

    """
    
    channels_in = 5
    n_class = 1024
    channels_out=4
    def __init__(self):
        print('')
        
    def _load_data_and_label(self):
    
        #be careful here it's one sample of the data so "one image of 5 channels"
        data, label = self._next_data()
        
                
        nx = data.shape[1]#X dimension
        ny = data.shape[0]#Y dimension
        return data.reshape(1, ny, nx, self.channels_in), label.reshape(1, ny, nx, self.channels_out)

    
    def __call__(self, n):#n here is the size of the minibatch
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels_in))
        Y = np.zeros((n, nx, ny, self.channels_out))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n): #again n not included
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y
    
class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays. 
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels_in]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.

    :param data: data numpy array. Shape=[n, X, Y, channels_in]
    :param label: label numpy array. Shape=[n, X, Y,classes]

    
    """
    
    def __init__(self, data, label, channels_in=5,channels_out=4, n_class = 1024):
        super(SimpleDataProvider, self).__init__()
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = n_class
        self.channels_in = channels_in
        self.channels_out=channels_out
       

    def _next_data(self):
        idx = np.random.choice(self.file_count)#random choice !! minibatch but file count is ALL THE IMAGES POSSIBLE 
        return self.data[idx], self.label[idx]    

