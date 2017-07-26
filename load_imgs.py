import tensorflow as tf
import h5py
import numpy as np
import cv2

def read_data(path):
    
    
    pan_tiff = cv2.imread(path+"14s14e32_c50y11.tif",cv2.IMREAD_UNCHANGED)
    pxs_tiff = cv2.imread(path+"PAN_14s14e32_c50y11.tif",cv2.IMREAD_UNCHANGED)
    xs1_tiff = cv2.imread(path+"XS1_14s14e32_c50y11.tif",cv2.IMREAD_UNCHANGED)
    xs2_tiff = cv2.imread(path+"XS2_14s14e32_c50y11.tif",cv2.IMREAD_UNCHANGED)
    xs3_tiff = cv2.imread(path+"XS3_14s14e32_c50y11.tif" ,cv2.IMREAD_UNCHANGED)
    xs4_tiff = cv2.imread(path+"XS4_14s14e32_c50y11.tif" ,cv2.IMREAD_UNCHANGED)
    
    
    return pan_tiff,pxs_tiff,xs1_tiff,xs2_tiff,xs3_tiff,xs4_tiff



def preprocess(path, scale, w,h,stride):
    
    pan_tiff,pxs_tiff,xs1_tiff,xs2_tiff,xs3_tiff,xs4_tiff=read_data(path)
    pxs_tiff=pxs_tiff/255.
    xs1_tiff=xs1_tiff/255.
    xs2_tiff=xs2_tiff/255.
    xs3_tiff=xs3_tiff/255.
    xs4_tiff=xs4_tiff/255.
    
    size_hr=[pxs_tiff.shape[0],pxs_tiff.shape[1]]
    size_lr=[xs1_tiff.shape[0],xs1_tiff.shape[1]]
    
    new_height = int(round(xs1_tiff.shape[0] * scale))
    new_width = int(round(xs1_tiff.shape[1] * scale))
    

    with tf.Session() as sess:
        
     
        xs1_tiff=tf.reshape(xs1_tiff,[1,size_lr[0],size_lr[1],1])
        xs2_tiff=tf.reshape(xs2_tiff,[1,size_lr[0],size_lr[1],1])
        xs3_tiff=tf.reshape(xs3_tiff,[1,size_lr[0],size_lr[1],1])
        xs4_tiff=tf.reshape(xs4_tiff,[1,size_lr[0],size_lr[1],1])
        pxs_tiff=tf.reshape(pxs_tiff,[1,size_hr[0],size_hr[1],1])
        
        
        xs1_hr=tf.image.resize_images(xs1_tiff, [new_height, new_width])
        xs2_hr=tf.image.resize_images(xs2_tiff, [new_height, new_width])
        xs3_hr=tf.image.resize_images(xs3_tiff, [new_height, new_width])
        xs4_hr=tf.image.resize_images(xs4_tiff, [new_height, new_width])
        
        xs_hr=tf.concat((xs1_hr,xs2_hr,xs3_hr,xs4_hr),axis=-1)
        xs_hr=tf.slice(xs_hr,[0,0,0,0],[-1,size_hr[0],size_hr[1],-1])
        xs_hr = tf.cast(xs_hr,tf.float64)
   
        
        input_=  tf.concat((pxs_tiff,xs_hr),axis=-1)
        label_=tf.reshape(pan_tiff,[1,pan_tiff.shape[0],pan_tiff.shape[1],pan_tiff.shape[2]])
    
        
        size_input=input_.get_shape().as_list()
        size_label=label_.get_shape().as_list()
       
        label_ = tf.cast(label_,tf.float64)

        tot_=tf.concat((input_,label_),axis=-1)
        tot_patches=tf.extract_image_patches(images=tot_,ksizes=[1,w,h,1],strides=[1,stride[0],stride[1],1],rates=[1,1,1,1],padding='VALID')
        size_tot_patches=tot_patches.get_shape().as_list()
        tot_patches=tf.reshape(tot_patches,[size_tot_patches[1]*size_tot_patches[2],w,h,size_input[-1]+size_label[-1]])
        size_tot_patches=tot_patches.get_shape().as_list()
        
        tot_patches=tf.random_shuffle(tot_patches)
        #input_patches=tf.slice(tot_patches,[0,0,0,0],[size_tot_patches[0],size_tot_patches[1],size_tot_patches[2],size_input[-1]]).eval()
        #label_patches=tf.slice(tot_patches,[0,0,0,size_input[-1]],[size_tot_patches[0],size_tot_patches[1],size_tot_patches[2],-1]).eval()
        tot_patches=tot_patches.eval()
        input_patches=tot_patches[:,:,:,0:size_input[-1]]
        label_patches=tot_patches[:,:,:,size_input[-1]:size_input[-1]+size_label[-1]]
                
        
    return input_patches,label_patches

def prepare_data(path,input_patches,label_patches,p_train):
    
    sub_input_sequence_train = []
    sub_label_sequence_train = []
    
    sub_input_sequence_test = []
    sub_label_sequence_test = []
  
    size_dataset=input_patches.shape[0]
    size_train=int(round(p_train*size_dataset))
    
    for idx in range(0,size_train):
        
        sub_input=input_patches[idx]
        sub_label=label_patches[idx]
        
        sub_input_sequence_train.append(sub_input)
        sub_label_sequence_train.append(sub_label)

    for idx in range(size_train,size_dataset):
        
        sub_input=input_patches[idx]
        sub_label=label_patches[idx]
        
        sub_input_sequence_test.append(sub_input)
        sub_label_sequence_test.append(sub_label)
    
    arrdata_input_train = np.asarray(sub_input_sequence_train)
    arrdata_label_train = np.asarray(sub_label_sequence_train)
		
	
    arrdata_input_test= np.asarray(sub_input_sequence_test)
    arrdata_label_test= np.asarray(sub_label_sequence_test)
    make_data( path,arrdata_input_train, arrdata_label_train,arrdata_input_test,arrdata_label_test)
        
def make_data( path,data_train, label_train,data_test,label_test):
	
    savepath_train = path+ 'train.h5'
    savepath_test = path+'test.h5'
    with h5py.File(savepath_train, 'w') as hf:
        hf.create_dataset('data', data=data_train)
        hf.create_dataset('label', data=label_train)   

    with h5py.File(savepath_test, 'w') as hf:
        hf.create_dataset('data', data=data_test)
        hf.create_dataset('label', data=label_test)
