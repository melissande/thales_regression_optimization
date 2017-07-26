import h5py
import numpy as np

def extract(path,batch_train,batch_test):
    
    savepath_train = path+ '/train.h5'
    savepath_test = path+'/test.h5'
    
    with h5py.File(savepath_train, 'r') as hf:
        data_train =np.array(hf.get('data'))
        label_train =np.array(hf.get('label'))
        if batch_train!=0:
            data_train=data_train[0:batch_train,:,:,:]
            label_train=label_train[0:batch_train,:,:,:]
        

    with h5py.File(savepath_test, 'r') as hf:
        data_test=  np.array(hf.get('data'))
        label_test =np.array(hf.get('label'))
        if batch_test!=0:
            data_test=data_test[0:batch_test,:,:,:]
            label_test=label_test[0:batch_test,:,:,:]

    data_train-=np.amin(data_train)
    data_train=np.true_divide(data_train,np.amax(data_train),casting='unsafe')
    
    data_test-=np.amin(data_test)
    data_test=np.true_divide(data_test,np.amax(data_test),casting='unsafe')
    
    label_train-=np.amin(label_train)
    label_train=np.true_divide(label_train,np.amax(label_train),casting='unsafe')

    label_test-=np.amin(label_test)
    label_test=np.true_divide(label_test,np.amax(label_test),casting='unsafe')
    return data_train,label_train,data_test,label_test
