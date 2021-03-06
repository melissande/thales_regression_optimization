from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

import util
from layers import (weight_variable, weight_variable_devonc, bias_variable, 
                            conv2d, deconv2d, max_pool, crop_and_concat,
                            l2_loss)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_conv_net(x, keep_prob, channels_in, channels_out,n_class, layers=3, features_root=32, filter_size=3, pool_size=2, summaries=True):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,channels_in]
    :param keep_prob: dropout probability tensor
    :param channels_in: number of channels in the input image
    :param channels_out: number of channels in the output image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    
    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels_in]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]
 
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    
    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, layers):
        features = 2**layer*features_root
        stddev =1.5* np.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, channels_in, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
            
        w2 = weight_variable([filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])
        
        conv1 = conv2d(in_node, w1, keep_prob)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
        
        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        size -= 4 
        if layer < layers-1:#because after it's the end of the U
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2
        
    in_node = dw_h_convs[layers-1]#it's the last layer the bottom of the U but it's because
                                    #of the definition of range we have layers -1 and not layers
        
    # up layers
    for layer in range(layers-2, -1, -1):#we don't begin at the bottom of the U
        features = 2**(layer+1)*features_root
        stddev = 1.5*np.sqrt(2 / (filter_size**2 * features))
        
        wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
        bd = bias_variable([features//2]) # weights and bias for upsampling 
                                            #from a layer to another !!
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd) 
                                        #recall that in_node is the last layer
                                        #bottom of the U
        
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)#layer
                                                         #before the bottom of the  U 
        deconv[layer] = h_deconv_concat
        
        w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
        w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
        b1 = bias_variable([features//2])
        b2 = bias_variable([features//2])
        
        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(h_conv, w2, keep_prob)
        in_node = tf.nn.relu(conv2 + b2)
        up_h_convs[layer] = in_node

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        size *= 2
        size -= 4

    # Output Map

    weight = weight_variable([1, 1, features_root, channels_out], stddev)
    bias = bias_variable([channels_out])
    conv = conv2d(in_node, weight, tf.constant(1.0))
    output_map = conv + bias
    up_h_convs["out"] = output_map
    
    if summaries:
        for i, (c1, c2) in enumerate(convs):
            tf.summary.image('summary_conv_%02d_01'%i, get_image_summary(c1))
            tf.summary.image('summary_conv_%02d_02'%i, get_image_summary(c2))
            
        for k in pools.keys():
            tf.summary.image('summary_pool_%02d'%k, get_image_summary(pools[k]))
        
        for k in deconv.keys():
            tf.summary.image('summary_deconv_concat_%02d'%k, get_image_summary(deconv[k]))
            
        for k in dw_h_convs.keys():
            tf.summary.histogram("dw_convolution_%02d"%k + '/activations', dw_h_convs[k])

        for k in up_h_convs.keys():
            tf.summary.histogram("up_convolution_%s"%k + '/activations', up_h_convs[k])
            
    variables = []
    for w1,w2 in weights:
        variables.append(w1)
        variables.append(w2)
        
    for b1,b2 in biases:
        variables.append(b1)
        variables.append(b2)

    
    return output_map, variables, int(in_size - size)
def srcnn(x,keep_prob, channels_in,channels_out, layers =1,filters_nb=[64,32], filters_widths=[9,1,5], summaries=True):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,channels_in]
    :param channels_in: number of channels in the input image
    :param channels_out: number of channels in the output image
    :param filters_nb: number of filters for each layer
    :param filters_widths: size of the filter for each layer
    :param summaries: Flag if summaries should be created
    """
    
    logging.info("Layers {layers},Number of filters {filters_nb}, filter size {filters_widths}".format(layers=layers,filters_nb=filters_nb,
                                                                                                           filters_widths=filters_widths))
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels_in]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]

 
    weights = []
    biases = []
    convs = []
    h_convs = OrderedDict()
    filters_loop=np.hstack(([channels_in],filters_nb,[channels_out]))
    
    # Patch extraction and Non linear Mapping
    for layer in range(0,layers+2):
        stddev=np.sqrt(2/(filters_loop[layer+1]*filters_widths[layer]**2))
        w=weight_variable([filters_widths[layer], filters_widths[layer], filters_loop[layer], filters_loop[layer+1]], stddev)
        b = bias_variable([filters_loop[layer+1]])
        conv_ = conv2d(in_node, w,keep_prob)
        if layer<layers+1:
            h_convs[layer] = tf.nn.relu(conv_ + b)
            in_node = h_convs[layer]
        else:
            output_map=conv_+b
                
        weights.append(w)
        biases.append(b)
        convs.append(conv_)



    # Output Map
    
    h_convs["out"] = output_map
    
    if summaries:
        for i, c in enumerate(convs):
            tf.summary.image('summary_conv_%02d'%i, get_image_summary(c))
        
        for k in h_convs.keys():
            tf.summary.histogram("convolution_%s"%k + '/activations', h_convs[k])

        
            
    variables = []
    for w in weights:
        variables.append(w)

    for b in biases:
        variables.append(b)


    
    return output_map, variables

class Unet(object):
    """
    A unet implementation
    
    :param channels_in: (optional) number of channels in the input image
    :param channels_out: (optional) number of channels in the output image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'l2_loss'
    """
    
    def __init__(self, channels_in=5, channels_out=4,n_class=1024, cost="l2_loss", **kwargs):
        tf.reset_default_graph() #begin with a neutral graph
        

        self.n_class = n_class
        self.channels_out = channels_out
        self.summaries = kwargs.get("summaries", True)
        
        self.x = tf.placeholder("float", shape=[None, None, None, channels_in])
        self.y = tf.placeholder("float", shape=[None, None, None, channels_out])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        logits, self.variables,self.offset = create_conv_net(self.x, self.keep_prob, channels_in, channels_out, n_class, **kwargs)
       # logits, self.variables = srcnn(self.x, self.keep_prob, channels_in, channels_out, **kwargs)
        
        
        
                        
        self.l2_loss =  l2_loss(self.y,logits)
        self.gradients_node = tf.gradients(self.l2_loss, self.variables)
        
        self.predicter = logits
       
        
   

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels_in]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 

        """
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        with tf.Session(config=config) as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.channels_out))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

        return prediction

        
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location 
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)

class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """
    
    prediction_path = "prediction"
    verification_batch_size = 15
    
    def __init__(self, net, batch_size=4, optimizer="adam", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        
    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.07)#0.2
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.999)
            momentum = self.opt_kwargs.pop("momentum", 0.2)#0.2
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.l2_loss, 
                                                                                global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.00001)#0.001
            self.learning_rate_node = tf.Variable(learning_rate)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                               **self.opt_kwargs).minimize(self.net.l2_loss,
                                                                     global_step=global_step)
        
        return optimizer
        
    def _initialize(self, training_iters, output_path, restore):
        global_step = tf.Variable(0)
        
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))
        
        if self.net.summaries:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        
        tf.summary.scalar('l2_loss_valid', self.net.l2_loss)
 
        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()        
        self.loss_verif=tf.summary.scalar('l2_loss_verif', self.net.l2_loss)
 
        init = tf.global_variables_initializer()
        
        self.prediction_path = os.path.abspath(output_path)
        output_path = os.path.abspath(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(self.prediction_path))
            shutil.rmtree(self.prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(self.prediction_path):
            logging.info("Allocating '{:}'".format(self.prediction_path))
            os.makedirs(self.prediction_path)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
        
        return init

    def train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1, restore=False, write_graph=False):
        """
        Lauches the training process
        
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        
        init = self._initialize(training_iters, output_path, restore)
        
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)
            
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
            
            test_x, test_y = data_provider(self.verification_batch_size,verif=True) #verify the minibatch works
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            pred_shape = self.store_prediction(sess,summary_writer,-1, test_x, test_y, "_init")
            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")
            
            avg_gradients = None
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)
                     
                    # Run optimization op (backprop)
                    _, loss, lr, gradients = sess.run((self.optimizer, self.net.l2_loss, self.learning_rate_node, self.net.gradients_node), 
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                                 self.net.keep_prob: dropout})

                    if avg_gradients is None:
                        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
                    for i in range(len(gradients)):
                        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step+1)))) + (gradients[i] / (step+1))
                        
                    norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                    self.norm_gradients_node.assign(norm_gradients).eval()
                    
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x, util.crop_to_shape(batch_y, pred_shape))
                        
                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess,summary_writer, epoch,test_x, test_y, "epoch_%s"%epoch)
                    
                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished!")
            
            return save_path
        
    def store_prediction(self, sess, summary_writer,epoch,batch_x, batch_y,  name):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x, 
                                                             self.net.y: batch_y, 
                                                             self.net.keep_prob: 1.})
        pred_shape = prediction.shape
        summary_loss_verif,loss = sess.run([self.loss_verif,self.net.l2_loss], feed_dict={self.net.x: batch_x, 
                                                       self.net.y: util.crop_to_shape(batch_y, pred_shape), 
                                                       self.net.keep_prob: 1.})
        
        summary_writer.add_summary(summary_loss_verif,epoch)
        summary_writer.flush()
        logging.info(" loss= {:.4f}".format(loss))
         #modified through reshape
        batch_x=util.to_rgb(util.crop_to_shape(batch_x, pred_shape))
        batch_y=util.to_rgb(util.crop_to_shape(batch_y, pred_shape))
        prediction=util.to_rgb(prediction)
        for i in range(batch_y.shape[0]):
            if name=='_init':
                util.save_image(batch_x[i,:,:,0], "%s/data_%s_%s0.jpg"%(self.prediction_path, name,i))
                util.save_image(batch_y[i,:,:,:], "%s/true_%s_%s0.jpg"%(self.prediction_path, name,i))
            util.save_image(prediction[i,:,:,:], "%s/pred_%s_%s0.jpg"%(self.prediction_path, name,i))
        #img = util.combine_img_prediction(batch_x, batch_y, prediction)
        #util.save_image(img, "%s/%s.jpg"%(self.prediction_path, name))
        
        return pred_shape

    def compute_pred_loss(self, prediction, label):
        return np.mean(np.square(y_- output_map))                        
    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))
    

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, predictions = sess.run([self.summary_op, 
                                                            self.net.l2_loss, 
                                                            self.net.predicter], 
                                                           feed_dict={self.net.x: batch_x,
                                                                      self.net.y: batch_y,
                                                                      self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss= {:.4f}".format(step,loss))




def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V

