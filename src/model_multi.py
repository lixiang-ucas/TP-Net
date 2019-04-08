#coding: utf-8

from __future__ import division
# import tensorflow as tf
import sugartensor as tf
from tensorflow.python.ops import variable_scope as vs
import numpy as np
import math
import time
import h5py
# random_seed = 234
import tf_util

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    
    with tf.variable_scope('tnet', reuse=tf.AUTO_REUSE) as vs:
        # vs.reuse_variables()
        
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value

        input_image = tf.expand_dims(point_cloud, -1)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='tconv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='tconv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='tconv3', bn_decay=bn_decay)
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='tmaxpool')

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='tfc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='tfc2', bn_decay=bn_decay)

        with tf.variable_scope('transform_XYZ') as sc:
            assert(K==3)
            weights = tf.get_variable('weights', [256, 3*K],
                                      initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            biases = tf.get_variable('biases', [3*K],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [batch_size, 3, K])
        return transform

def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1) #, seed=random_seed
#        return tf.Variable(initial)
		return tf.get_variable("weights", initializer=initial, regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))
    

def bias_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1) #, seed=random_seed
#        return tf.Variable(initial)
		return tf.get_variable("bias",  initializer=initial)

def assignment_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1) #, seed=random_seed
#        return tf.Variable(initial)
		return tf.get_variable("assignment",  initializer=initial)

def tile_repeat(n, repTime):
		'''
		create something like 111..122..2333..33 ..... n..nn 
		one particular number appears repTime consecutively.
		This is for flattening the indices.
		'''
		idx = tf.range(n)
		idx = tf.reshape(idx, [-1, 1])    # Convert to a n x 1 matrix.
		idx = tf.tile(idx, [1, repTime])  # Create multiple columns, each column has one number repeats repTime 
		y = tf.reshape(idx, [-1])
		return y

def get_weight_assigments(x, adj, u, v, c):
		batch_size, in_channels, num_points = x.get_shape().as_list()
		batch_size, num_points, K = adj.get_shape().as_list()
		M, in_channels = u.get_shape().as_list()
		# [batch_size, M, N]
		ux = tf.map_fn(lambda x: tf.matmul(u, x), x)
		vx = tf.map_fn(lambda x: tf.matmul(v, x), x)
		# [batch_size, N, M]
		vx = tf.transpose(vx, [0, 2, 1])
		# [batch_size, N, K, M]
		patches = get_patches(vx, adj)
		# [K, batch_size, M, N]
		patches = tf.transpose(patches, [2, 0, 3, 1])
		# [K, batch_size, M, N]
		patches = tf.add(ux, patches)
		# [K, batch_size, N, M]
		patches = tf.transpose(patches, [0, 1, 3, 2])
		patches = tf.add(patches, c)
		# [batch_size, N, K, M]
		patches = tf.transpose(patches, [1, 2, 0, 3])
		patches = tf.nn.softmax(patches)
		return patches

def get_weight_assigments_translation_invariance(x, adj, u, c):
		batch_size, num_points, in_channels = x.get_shape().as_list()
		batch_size, num_points, K = adj.get_shape().as_list()
		M, in_channels = u.get_shape().as_list()
		# [batch, N, K, ch]
		patches = get_patches(x, adj)
		# [batch, N, ch, 1]
		x = tf.reshape(x, [-1, num_points, in_channels, 1])
		# [batch, N, ch, K]
		patches = tf.transpose(patches, [0, 1, 3, 2])
		# [batch, N, ch, K]
		patches = tf.subtract(x, patches)
		# [batch, ch, N, K]
		patches = tf.transpose(patches, [0, 2, 1, 3])
		# [batch, ch, N*K]
		x_patches = tf.reshape(patches, [-1, in_channels, num_points*K])
		# batch, M, N*K
		patches = tf.map_fn(lambda x: tf.matmul(u, x) , x_patches)
		# batch, M, N, K
		patches = tf.reshape(patches, [-1, M, num_points, K])
		# [batch, K, N, M]
		patches = tf.transpose(patches, [0, 3, 2, 1])
		# [batch, K, N, M]
		patches = tf.add(patches, c)
		# batch, N, K, M
		patches = tf.transpose(patches, [0, 2, 1, 3])
		patches = tf.nn.softmax(patches)
		return patches

def get_slices(x, adj):
		batch_size, num_points, in_channels = x.get_shape().as_list()
		batch_size, input_size, K = adj.get_shape().as_list()
		zeros = tf.zeros([batch_size, 1, in_channels], dtype=tf.float32)
		x = tf.concat([zeros, x], 1)
		x = tf.reshape(x, [batch_size*(num_points+1), in_channels])
		adj = tf.reshape(adj, [batch_size*num_points*K])
		adj_flat = tile_repeat(batch_size, num_points*K)
		adj_flat = adj_flat*(num_points+1)
		adj_flat = adj_flat + adj
		adj_flat = tf.reshape(adj_flat, [batch_size*num_points, K])
		slices = tf.gather(x, adj_flat)
		slices = tf.reshape(slices, [batch_size, num_points, K, in_channels])
		return slices

def get_patches(x, adj):
		batch_size, num_points, in_channels = x.get_shape().as_list()
		batch_size, num_points, K = adj.get_shape().as_list()
		patches = get_slices(x, adj)
		return patches

def custom_conv2d(x, adj, out_channels, M, translation_invariance=True):
    with tf.variable_scope('custom_conv2d'):
    		if translation_invariance == False:
    				batch_size, input_size, in_channels = x.get_shape().as_list()
    				W = weight_variable([M, out_channels, in_channels])
    				b = bias_variable([out_channels])
    				u = assignment_variable([M, in_channels])
    				v = assignment_variable([M, in_channels])
    				c = assignment_variable([M])
    				batch_size, input_size, K = adj.get_shape().as_list()
    				# Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
    				adj_size = tf.count_nonzero(adj, 2)
    				#deal with unconnected points: replace NaN with 0
    				non_zeros = tf.not_equal(adj_size, 0)
    				adj_size = tf.cast(adj_size, tf.float32)
    				adj_size = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
    				# [batch_size, input_size, 1, 1]
    				adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
    				# [batch_size, in_channels, input_size]
    				x = tf.transpose(x, [0, 2, 1])
    				W = tf.reshape(W, [M*out_channels, in_channels])
    				# Multiple w and x -> [batch_size, M*out_channels, input_size]
    				wx = tf.map_fn(lambda x: tf.matmul(W, x), x)
    				# Reshape and transpose wx into [batch_size, input_size, M*out_channels]
    				wx = tf.transpose(wx, [0, 2, 1])
    				# Get patches from wx - [batch_size, input_size, K(neighbours-here input_size), M*out_channels]
    				patches = get_patches(wx, adj)
    				# [batch_size, input_size, K, M]
    				q = get_weight_assigments(x, adj, u, v, c)
    				# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
    				patches = tf.reshape(patches, [batch_size, input_size, K, M, out_channels])
    				# [out, batch_size, input_size, K, M]
    				patches = tf.transpose(patches, [4, 0, 1, 2, 3])
    				patches = tf.multiply(q, patches)
    				patches = tf.transpose(patches, [1, 2, 3, 4, 0])
    				# Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
    				patches = tf.reduce_sum(patches, axis=2)
    				patches = tf.multiply(adj_size, patches)
    				# Add add elements for all m
    				patches = tf.reduce_sum(patches, axis=2)
    				# [batch_size, input_size, out]
    				patches = patches + b
    				return patches
    		else:
    				print("Translation-invariant\n")
    				batch_size, input_size, in_channels = x.get_shape().as_list()
    				W = weight_variable([M, out_channels, in_channels])
    				tf.add_to_collection('my_reg', W)
    				b = bias_variable([out_channels])
    				with tf.variable_scope('u'):
    					u = assignment_variable([M, in_channels])
    				with tf.variable_scope('c'):
    					c = assignment_variable([M])
    				batch_size, input_size, K = adj.get_shape().as_list()
    				# Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
    				adj_size = tf.count_nonzero(adj, 2)
    				#deal with unconnected points: replace NaN with 0
    				non_zeros = tf.not_equal(adj_size, 0)
    				adj_size = tf.cast(adj_size, tf.float32)
    				adj_size = tf.where(non_zeros,tf.reciprocal(adj_size),tf.zeros_like(adj_size))
    				# [batch_size, input_size, 1, 1]
    				adj_size = tf.reshape(adj_size, [batch_size, input_size, 1, 1])
    				# [batch_size, input_size, K, M]
    				q = get_weight_assigments_translation_invariance(x, adj, u, c)
    				# [batch_size, in_channels, input_size]
    				x = tf.transpose(x, [0, 2, 1])
    				W = tf.reshape(W, [M*out_channels, in_channels])
    				# Multiple w and x -> [batch_size, M*out_channels, input_size]
    				wx = tf.map_fn(lambda x: tf.matmul(W, x), x)
    				# Reshape and transpose wx into [batch_size, input_size, M*out_channels]
    				wx = tf.transpose(wx, [0, 2, 1])
    				# Get patches from wx - [batch_size, input_size, K(neighbours-here input_size), M*out_channels]
    				patches = get_patches(wx, adj)
    				# [batch_size, input_size, K, M]
    				#q = get_weight_assigments_translation_invariance(x, adj, u, c)
    				# Element wise multiplication of q and patches for each input -- [batch_size, input_size, K, M, out]
    				patches = tf.reshape(patches, [batch_size, input_size, K, M, out_channels])
    				# [out, batch_size, input_size, K, M]
    				patches = tf.transpose(patches, [4, 0, 1, 2, 3])
    				patches = tf.multiply(q, patches)
    				patches = tf.transpose(patches, [1, 2, 3, 4, 0])
    				# Add all the elements for all neighbours for a particular m sum_{j in N_i} qwx -- [batch_size, input_size, M, out]
    				patches = tf.reduce_sum(patches, axis=2)
    				patches = tf.multiply(adj_size, patches)
    				# Add add elements for all m
    				patches = tf.reduce_sum(patches, axis=2)
    				# [batch_size, input_size, out]
    				patches = patches + b
    				return patches

def custom_lin(input, out_channels):
     with tf.variable_scope('custom_lin'):
    		batch_size, input_size, in_channels = input.get_shape().as_list()
    		W = weight_variable([in_channels, out_channels])
    		tf.add_to_collection('my_reg', W)
    		b = bias_variable([out_channels])
    		return tf.map_fn(lambda x: tf.matmul(x, W), input) + b

def custom_max_pool(input, kernel_size, stride=[2, 2], padding='VALID'):
		kernel_h, kernel_w = kernel_size
		stride_h, stride_w = stride
		outputs = tf.nn.max_pool(input, ksize=[1, kernel_h, kernel_w, 1], strides=[1, stride_h, stride_w, 1], padding=padding)
		return outputs
    
    
def un_max_pooling_1d(input, out_size):
    batch_size, input_size, in_channels = input.get_shape().as_list()
    
    outputs=tf.tile(input,[1,1,2])
    outputs=tf.reshape(outputs,[batch_size,input_size*2,in_channels])
#    outputs =np.zeros([batch_size, 1, in_channels])
#    
#    for i in range(input_size):        
#        outputs=tf.concat([outputs,tf.reshape(input[:,i,:],[batch_size,1,in_channels])],1)
#        if 2*i+1<out_size:            
#             outputs=tf.concat([outputs,tf.reshape(input[:,i,:],[batch_size,1,in_channels])],1)
#    outputs=tf.convert_to_tensor(outputs)     
    
    
    return outputs[:,:out_size,:]
#    return outputs[:,1:,:]
    
def un_max_pooling_1d_2(input, out_size):
    batch_size, input_size, in_channels = input.get_shape().as_list()
    
    up_padding =tf.zeros([batch_size, input_size, in_channels])
    outputs=tf.concat([input,up_padding],2)
    outputs=tf.reshape(outputs,[batch_size,input_size*2,in_channels])
    return outputs[:,:out_size,:]
    
def perm_tensor(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    # print x
    # print indices
    
    batch0=tf.gather(x[0], [indices[0]])
    batch1=tf.gather(x[1], [indices[1]])
    x_permute=tf.concat([batch0,batch1],axis=0)

    return x_permute

def wen_max_pool(x, indices):
    batch_size, point_size, channels = x.get_shape().as_list()
    output_size=indices[0].shape[0]
    
    x_permute = perm_tensor(x, indices)
    x_permute=tf.expand_dims(x_permute,axis=1)
    # print x_permute
    x_pool=custom_max_pool(x_permute,[1,2],[1,2])
    x_pool2=tf.squeeze(x_pool,axis=1)
    # print x_pool2
    return x_pool2
    
def get_bilinear_filter(filter_shape, upscale_factor):
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5
 
    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
 
    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                           shape=weights.shape)
    return bilinear_weights

def upsample_layer(bottom,n_channels, name, upscale_factor,output):
    kernel_size = 2*upscale_factor - upscale_factor%2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        # Shape of the bottom tensor
        in_shape = tf.shape(bottom)
     
        h = ((in_shape[1] - 1) * stride) + 1
        
#        if in_shape[2]%2==0:     
#            w = ((in_shape[2] - 1) * stride) + 2
#            print "w0"
#            print w
#        else:
#            w = ((in_shape[2] - 1) * stride) + 1
#            print "w1"
#            print w
        w=output
        new_shape = [in_shape[0], h, w, n_channels]
        output_shape = tf.stack(new_shape)
     
        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
     
        weights = get_bilinear_filter(filter_shape,upscale_factor)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')
     
    return deconv    

def get_model(x, adj, num_classes, architecture, Mconv=9, reuse=False):
    """ 
    0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
    """
#    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('FeastNet'):
    #     if reuse:                                        ### 改动部分 ###
    #         vs.get_variable_scope().reuse_variables()
        if architecture == 0:
            with tf.variable_scope('FC0'):
                out_channels_fc0 = 16
                h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
                print h_fc0
            # Conv1
            with tf.variable_scope('Conv1'):
                M_conv1 = Mconv
                out_channels_conv1 = 32
                h_conv1 = tf.nn.relu(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1))
                print h_conv1
            # Conv2
            with tf.variable_scope('Conv2'):
                M_conv2 = Mconv
                out_channels_conv2 = 64
                h_conv2 = tf.nn.relu(custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2))
                print h_conv2
            # Conv3
            with tf.variable_scope('Conv3'):
                M_conv3 = Mconv
                out_channels_conv3 = 128
                h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3))
                print h_conv3
            # Lin(1024)
            with tf.variable_scope('Lin0'):
                out_channels_fc1 = 1024
                h_fc1 = tf.nn.relu(custom_lin(h_conv3, out_channels_fc1))
                print h_fc1
            with tf.variable_scope('Lin1'):
            # Lin(num_classes)
                y_conv = custom_lin(h_fc1, num_classes)
                print y_conv
            return y_conv, h_fc0, h_fc1

# def get_model(x, adj, num_classes, architecture, Mconv=9, reuse=False):
#     """ 
#     0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
#     """
# #    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
#     with tf.variable_scope('FeastNet'):
#     #     if reuse:                                        ### 改动部分 ###
#     #         vs.get_variable_scope().reuse_variables()
#         if architecture == 0:
#             with tf.variable_scope('FC0'):
#                 out_channels_fc0 = 16
#                 h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
#                 print h_fc0
#             # Conv1
#             with tf.variable_scope('Conv1'):
#                 M_conv1 = Mconv
#                 out_channels_conv1 = 16 #32
#                 h_conv1 = tf.nn.relu(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1))
#                 print h_conv1
#             # Conv2
#             with tf.variable_scope('Conv2'):
#                 M_conv2 = Mconv
#                 out_channels_conv2 = 16 #64
#                 h_conv2 = tf.nn.relu(custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2))
#                 print h_conv2
#             # Conv3
#             with tf.variable_scope('Conv3'):
#                 M_conv3 = Mconv
#                 out_channels_conv3 = 16#128
#                 h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3))
#                 print h_conv3
#             # Lin(1024)
#             with tf.variable_scope('Lin0'):
#                 out_channels_fc1 = 256 #1024
#                 h_fc1 = tf.nn.relu(custom_lin(h_conv3, out_channels_fc1))
#                 print h_fc1
#             with tf.variable_scope('Lin1'):
#             # Lin(num_classes)
#                 y_conv = custom_lin(h_fc1, num_classes)
#                 print y_conv
#             return y_conv, h_fc0, h_fc1

def wen_permute_pool(x, adj, indices):
    batch_size, point_size, channels = x.get_shape().as_list()
    # output_size=indices[0].shape[0]
    # reorder
#    x_permute = perm_tensor(x, indices)
#    print x_permute
#    adj_t_float = tf.cast(adj, dtype=tf.float32)
#    adj_permute=perm_tensor(adj_t_float,indices)
#    print adj_permute
    
    x_permute=x
    adj_permute=adj
    x_pool = tf.nn.relu(tf.layers.max_pooling1d(x_permute,2,2,padding='same'))
     
    batch_size, point_size, channels = adj_permute.get_shape().as_list()
    adj_pool=tf.reshape(adj_permute,[batch_size,int(point_size/2),int(channels*2)])
    adj_pool=(adj_pool+1)/2
    adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                
#    x_pool = custom_pool(x_permute,tf.convert_to_tensor(range(0,output_size,2)))
#    adj_pool= custom_pool(adj_permute,tf.convert_to_tensor(range(0,output_size,2)))
    
    return x_pool,adj_pool

def get_model_multi(x, adj, num_classes, architecture, Mconv=9, reuse=False):
    """ 
    0 - input(N*3) - CONV(N*8) - POOL(N/2*8) - CONV(N/2*16) - POOL(N/4*16) - CONV(N/4*32) - CONV(N/4*16) - UP+CONCAT(N/2*32) - UP+CONCAT(N*16) - LIN(N*256)- Output(N*N)
    """
#    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('FeastNet'):
#         if reuse:                                        ### 改动部分 ###
#             vs.get_variable_scope().reuse_variables()
        if architecture == 0:
            
            B = x.get_shape()[0].value
            N = x.get_shape()[1].value
            print x
            
            
            # Conv0
            with tf.variable_scope('FC0'):
                M_conv0 = Mconv
                out_channels_conv0 = 8
                h_conv0 = tf.nn.relu(custom_conv2d(x, adj,out_channels_conv0,M_conv0))
                print "h_conv0:"
                print h_conv0
            
            # Pool0
            with tf.variable_scope('Pool0'):
#            h_pool0 = tf.nn.relu(wen_max_pool(h_conv0))
#                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
#                h_pool0 = tf.layers.max_pooling1d(h_conv0,2,2,padding='same')
#                adj_pool = tf.layers.max_pooling1d(adj,2,2,padding='same')
#                adj_pool = tf.nn.relu(tf.layers.max_pooling1d(adj,2,2,padding='same'))
#                h_pool0 = tf.nn.relu(wen_max_pool(h_conv0,x_perm))
#                adj=tf.cast(adj, dtype=tf.float32)
#                adj_pool = wen_max_pool(adj,x_perm)
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                
#                h_pool0,adj_pool = wen_permute_pool(h_conv0,adj,None)
               
                
                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
     
                batch_size, point_size, channels = adj.get_shape().as_list()
                if point_size%2==1:
                    adj_t=tf.concat([adj,tf.zeros([batch_size,1,channels],dtype=tf.int32)],axis=1)
                    print('lx debug', adj_t.shape)
                    batch_size, point_size, channels = adj_t.get_shape().as_list()
                    adj_pool=tf.reshape(adj_t,[batch_size,int(point_size/2),int(channels*2)])
                else:
                    adj_pool=tf.reshape(adj,[batch_size,int(point_size/2),int(channels*2)])
                adj_pool1=(adj_pool+1)/2
                adj_pool1=tf.cast(adj_pool1, dtype=tf.int32)
    
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                
#                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
##                adj_pool= custom_pool(adj,tf.convert_to_tensor(range(0,6890,2)))
#                batch_size, point_size, channels = adj.get_shape().as_list()
#                adj_pool=tf.reshape(adj,[batch_size,int(point_size/2),int(channels*2)])
#                adj_pool=(adj_pool+1)/2
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                print("h_pool0:", h_pool0)
                print('adj_pool1', adj_pool1)
            
            # Conv1
            with tf.variable_scope('Conv1'):
                M_conv1 = Mconv
                out_channels_conv1 = 16
                h_conv1 = tf.nn.relu(custom_conv2d(h_pool0, adj_pool, out_channels_conv1, M_conv1))
                print "h_conv1:"
                print h_conv1
            
            # Pool1
            with tf.variable_scope('Pool1'):
    #            h_pool0 = tf.nn.relu(wen_max_pool(h_conv0))
#                h_pool1 = tf.nn.relu(tf.layers.max_pooling1d(h_conv1,2,2,padding='same'))
##                h_pool1 = tf.layers.max_pooling1d(h_conv1,2,2,padding='same')
#                adj_pool2 = tf.layers.max_pooling1d(adj_pool,2,2,padding='same')
#                adj_pool2 = tf.nn.relu(tf.layers.max_pooling1d(adj_pool,2,2,padding='same'))
    
#                h_pool1 = custom_pool(h_conv1,tf.convert_to_tensor(range(0,3446,2)))
#                adj_pool2= custom_pool(adj_pool,tf.convert_to_tensor(range(0,3446,2)))
                
                h_pool1 = tf.nn.relu(tf.layers.max_pooling1d(h_conv1,2,2,padding='same'))
#                adj_pool2= custom_pool(adj_pool,tf.convert_to_tensor(range(0,3446,2)))
                batch_size, point_size, channels = adj_pool1.get_shape().as_list()
                if point_size%2==1:
                    adj_t=tf.concat([adj_pool1,tf.zeros([batch_size,1,channels],dtype=tf.int32)],axis=1)
                    print('lx debug', adj_t.shape)
                    batch_size, point_size, channels = adj_t.get_shape().as_list()
                    adj_pool=tf.reshape(adj_t,[batch_size,int(point_size/2),int(channels*2)]) #(2,3446,16)->(2,1722,24)
                else:
                    adj_pool=tf.reshape(adj_pool1,[batch_size,int(point_size/2),int(channels*2)])
                adj_pool2=(adj_pool+1)/2
                adj_pool2=tf.cast(adj_pool, dtype=tf.int32)
                print "h_pool1:"
                print h_pool1
                print adj_pool2
            
            # Conv2
            with tf.variable_scope('Conv2'):
                M_conv2 = Mconv
                out_channels_conv2 = 32
                h_conv2 = tf.nn.relu(custom_conv2d(h_pool1, adj_pool2, out_channels_conv2, M_conv2))
                print "h_conv2:"
                print h_conv2
            
            # Conv3
            with tf.variable_scope('Conv3'):
                M_conv3 = Mconv
                out_channels_conv3 = 16
                h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj_pool2, out_channels_conv3, M_conv3))
                print "h_conv3:"
                print h_conv3
            
            # UP0
            with tf.variable_scope('UP0'):
                
#                h_up0 = tf.nn.relu(wen_un_pool(h_conv3,3445))
                
#                h_up0 = tf.nn.relu(un_max_pooling_1d(h_conv3,3446))
                h_conv3=tf.reshape(h_conv3,[h_conv3.shape[0],1,h_conv3.shape[1],h_conv3.shape[2]])
#                print h_conv3
#                print out_channels_conv3
                
#                h_up0 = tf.layers.conv2d_transpose(h_conv3, out_channels_conv3, 2, strides=[ 1, 2], padding='same')
#                print h_up0
                h_up0 = upsample_layer(h_conv3, out_channels_conv3, "wen_deconv", 2,int((N+1)/2))#6571
#                print h_up0
                h_up0=tf.reshape(h_up0,[h_conv3.shape[0],int((N+1)/2),out_channels_conv3])
    #            adj_up0=un_max_pooling_1d(h_conv3,3445)
                print "h_up0:"
                print h_up0
            
            # Cont0  
            with tf.variable_scope('Cont0'):
                h_cont0 = tf.nn.relu(tf.concat([h_conv1,h_up0],2))
#                h_cont0 = tf.nn.relu(tf.concat([h_up0,h_conv1],2))
    #            adj_up0 = tf.cast(adj_up0, dtype=tf.int32)
    #            adj_cont0 = tf.concat([c,adj_up0],1)
                print "h_cont0:"
                print h_cont0
            
             # Conv4
            with tf.variable_scope('Conv4'):
                M_conv4 = Mconv
                out_channels_conv4 = 8
                h_conv4 = tf.nn.relu(custom_conv2d(h_cont0, adj_pool1, out_channels_conv4, M_conv4))
                print "h_conv4:"
                print h_conv4
            
             # UP1
            with tf.variable_scope('UP1'):
#                h_up1 = tf.nn.relu(wen_un_pool(h_conv4,6890))
                
#                h_up1 = tf.nn.relu(un_max_pooling_1d(h_conv4,6890))
                h_conv4=tf.reshape(h_conv4,[h_conv4.shape[0],1,h_conv4.shape[1],h_conv4.shape[2]])
#                print h_conv4
                h_up1 = upsample_layer(h_conv4, out_channels_conv4, "wen_deconv2", 2,N)#13141
                h_up1=tf.reshape(h_up1,[h_conv4.shape[0],N,out_channels_conv4])

#    #            adj_up0=un_max_pooling_1d(h_conv3,3445)
#                de_perm_up1=perm_tensor(h_up1,x_de_perm)
#                h_up1=de_perm_up1[:,:6890,:]
                print "h_up1:"
                print h_up1
            
            # Cont1  
            with tf.variable_scope('Cont1'):
                h_cont1 = tf.nn.relu(tf.concat([h_conv0,h_up1],2))
    #            adj_up0 = tf.cast(adj_up0, dtype=tf.int32)
    #            adj_cont0 = tf.concat([c,adj_up0],1)
                print "h_cont1:"
                print h_cont1
            
            # Lin(1024)
            with tf.variable_scope('Lin0'):
                out_channels_fc1 = 256 #1024
                h_fc1 = tf.nn.relu(custom_lin(h_cont1, out_channels_fc1))
                print h_fc1
            
            # Lin(num_classes)
            with tf.variable_scope('Lin1'):
                y_conv = custom_lin(h_fc1, num_classes)
                print y_conv
            return y_conv

def get_lleWb(X, nbr_idc):

    B = batch_size = X.get_shape()[0].value
    N = num_point = X.get_shape()[1].value
    D = X.get_shape()[2].value
    K = nbr_idc.get_shape()[2].value
    
    ind = tf.reshape(nbr_idc,[-1,1])
    ind = ind+tf.reshape(N*((tf.range(B*N*K)//(N*K))),[-1,1])
    ind_cen = tf.reshape(tf.range(B*N*K)//K,[-1,1])
    X = tf.reshape(X,[-1,D])
    z = tf.gather_nd(X,ind)
    zmi = tf.gather_nd(X,ind_cen)
    z = z - zmi
    z = tf.reshape(z,[B,N,K,D])

    tol=1e-3
    C = tf.matmul(z,tf.transpose(z,[0,1,3,2])) #[B,N,K,K]
    # C = C + tol*tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(K,dtype=tf.float32),0),0),[B,N,1,1])*tf.expand_dims(tf.expand_dims(tf.trace(C),-1),-1)
    C = tf.cast(C,tf.float64)
    b = tf.tile(tf.expand_dims(tf.expand_dims(tf.ones([K,1]),0),0),[B,N,1,1])
    b = tf.cast(b,tf.float64)
    W = tf.matrix_solve_ls(C,b,fast=True, l2_regularizer=1e-3)
    W = (W+1e-8)/(tf.reduce_sum(W,2,keep_dims=True)+1e-8)
    W = tf.squeeze(W) #[B,N,K]
    s_shape = [B*N*N,]

    ind = tf.reshape(nbr_idc,[-1,K])
    ind = ind+N*tf.reshape(tf.range(B*N*K)//K,[B*N,K])
    ind = tf.reshape(ind,[B*N*K,])
    
    W = tf.reshape(W,[B*N*K,])
    _,ord = tf.nn.top_k(-ind,B*N*K,True)
    W = tf.gather(W,ord)
    ind = tf.gather(ind,ord)
    
    WW = tf.sparse_to_dense(tf.cast(ind,tf.int64), s_shape, W, validate_indices=False)
    WW = tf.reshape(WW,[B,N,N])
    WW = tf.cast(WW,tf.float32)
    return WW, nbr_idc


def get_model_multi_lle(x, adj, num_classes, architecture, Mconv=9, reuse=False):
    """ 
    0 - input(N*3) - CONV(N*8) - POOL(N/2*8) - CONV(N/2*16) - POOL(N/4*16) - CONV(N/4*32) - CONV(N/4*16) - UP+CONCAT(N/2*32) - UP+CONCAT(N*16) - LIN(N*256)- Output(N*N)
    """
#    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('FeastNet'):
#         if reuse:                                        ### 改动部分 ###
#             vs.get_variable_scope().reuse_variables()
        if architecture == 0:
            
            B = x.get_shape()[0].value
            N = x.get_shape()[1].value
            print x
            # Conv0
            with tf.variable_scope('FC0'):
                M_conv0 = Mconv
                out_channels_conv0 = 8
                h_conv0 = tf.nn.relu(custom_conv2d(x, adj,out_channels_conv0,M_conv0))
                print "h_conv0:"
                print h_conv0
            
            # Pool0
            with tf.variable_scope('Pool0'):
#            h_pool0 = tf.nn.relu(wen_max_pool(h_conv0))
#                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
#                h_pool0 = tf.layers.max_pooling1d(h_conv0,2,2,padding='same')
#                adj_pool = tf.layers.max_pooling1d(adj,2,2,padding='same')
#                adj_pool = tf.nn.relu(tf.layers.max_pooling1d(adj,2,2,padding='same'))
#                h_pool0 = tf.nn.relu(wen_max_pool(h_conv0,x_perm))
#                adj=tf.cast(adj, dtype=tf.float32)
#                adj_pool = wen_max_pool(adj,x_perm)
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                
#                h_pool0,adj_pool = wen_permute_pool(h_conv0,adj,None)
               
                
                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
     
                batch_size, point_size, channels = adj.get_shape().as_list()
                if point_size%2==1:
                    adj_t=tf.concat([adj,tf.zeros([batch_size,1,channels],dtype=tf.int32)],axis=1)
                    print('lx debug', adj_t.shape)
                    batch_size, point_size, channels = adj_t.get_shape().as_list()
                    adj_pool=tf.reshape(adj_t,[batch_size,int(point_size/2),int(channels*2)])
                else:
                    adj_pool=tf.reshape(adj,[batch_size,int(point_size/2),int(channels*2)])
                adj_pool1=(adj_pool+1)/2
                adj_pool1=tf.cast(adj_pool1, dtype=tf.int32)
    
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                
#                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
##                adj_pool= custom_pool(adj,tf.convert_to_tensor(range(0,6890,2)))
#                batch_size, point_size, channels = adj.get_shape().as_list()
#                adj_pool=tf.reshape(adj,[batch_size,int(point_size/2),int(channels*2)])
#                adj_pool=(adj_pool+1)/2
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                print("h_pool0:", h_pool0)
                print('adj_pool1', adj_pool1)
            
            # Conv1
            with tf.variable_scope('Conv1'):
                M_conv1 = Mconv
                out_channels_conv1 = 16
                h_conv1 = tf.nn.relu(custom_conv2d(h_pool0, adj_pool1, out_channels_conv1, M_conv1))
                print "h_conv1:"
                print h_conv1
            
            # Pool1
            with tf.variable_scope('Pool1'):
    #            h_pool0 = tf.nn.relu(wen_max_pool(h_conv0))
#                h_pool1 = tf.nn.relu(tf.layers.max_pooling1d(h_conv1,2,2,padding='same'))
##                h_pool1 = tf.layers.max_pooling1d(h_conv1,2,2,padding='same')
#                adj_pool2 = tf.layers.max_pooling1d(adj_pool,2,2,padding='same')
#                adj_pool2 = tf.nn.relu(tf.layers.max_pooling1d(adj_pool,2,2,padding='same'))
    
#                h_pool1 = custom_pool(h_conv1,tf.convert_to_tensor(range(0,3446,2)))
#                adj_pool2= custom_pool(adj_pool,tf.convert_to_tensor(range(0,3446,2)))
                
                h_pool1 = tf.nn.relu(tf.layers.max_pooling1d(h_conv1,2,2,padding='same'))
#                adj_pool2= custom_pool(adj_pool,tf.convert_to_tensor(range(0,3446,2)))
                batch_size, point_size, channels = adj_pool1.get_shape().as_list()
                if point_size%2==1:
                    adj_t=tf.concat([adj_pool1,tf.zeros([batch_size,1,channels],dtype=tf.int32)],axis=1)
                    print('lx debug', adj_t.shape)
                    batch_size, point_size, channels = adj_t.get_shape().as_list()
                    adj_pool=tf.reshape(adj_t,[batch_size,int(point_size/2),int(channels*2)]) #(2,3446,16)->(2,1722,24)
                else:
                    adj_pool=tf.reshape(adj_pool1,[batch_size,int(point_size/2),int(channels*2)])
                adj_pool2=(adj_pool+1)/2
                adj_pool2=tf.cast(adj_pool, dtype=tf.int32)
                print "h_pool1:"
                print h_pool1
                print adj_pool2
            
            # Conv2
            with tf.variable_scope('Conv2'):
                M_conv2 = Mconv
                out_channels_conv2 = 32
                h_conv2 = tf.nn.relu(custom_conv2d(h_pool1, adj_pool2, out_channels_conv2, M_conv2))
                print "h_conv2:"
                print h_conv2
            
            # Conv3
            with tf.variable_scope('Conv3'):
                M_conv3 = Mconv
                out_channels_conv3 = 16
                h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj_pool2, out_channels_conv3, M_conv3))
                print "h_conv3:"
                print h_conv3
            
            # UP0
            with tf.variable_scope('UP0'):
                
#                h_up0 = tf.nn.relu(wen_un_pool(h_conv3,3445))
                
#                h_up0 = tf.nn.relu(un_max_pooling_1d(h_conv3,3446))
                h_conv3=tf.reshape(h_conv3,[h_conv3.shape[0],1,h_conv3.shape[1],h_conv3.shape[2]])
#                print h_conv3
#                print out_channels_conv3
                
#                h_up0 = tf.layers.conv2d_transpose(h_conv3, out_channels_conv3, 2, strides=[ 1, 2], padding='same')
#                print h_up0
                h_up0 = upsample_layer(h_conv3, out_channels_conv3, "wen_deconv", 2,int((N+1)/2))#6571
#                print h_up0
                h_up0=tf.reshape(h_up0,[h_conv3.shape[0],int((N+1)/2),out_channels_conv3])
    #            adj_up0=un_max_pooling_1d(h_conv3,3445)
                print "h_up0:"
                print h_up0
            
            # Cont0  
            with tf.variable_scope('Cont0'):
                h_cont0 = tf.nn.relu(tf.concat([h_conv1,h_up0],2))
#                h_cont0 = tf.nn.relu(tf.concat([h_up0,h_conv1],2))
    #            adj_up0 = tf.cast(adj_up0, dtype=tf.int32)
    #            adj_cont0 = tf.concat([c,adj_up0],1)
                print "h_cont0:"
                print h_cont0
            
             # Conv4
            with tf.variable_scope('Conv4'):
                M_conv4 = Mconv
                out_channels_conv4 = 8
                h_conv4 = tf.nn.relu(custom_conv2d(h_cont0, adj_pool1, out_channels_conv4, M_conv4))
                print "h_conv4:"
                print h_conv4
            
             # UP1
            with tf.variable_scope('UP1'):
#                h_up1 = tf.nn.relu(wen_un_pool(h_conv4,6890))
                
#                h_up1 = tf.nn.relu(un_max_pooling_1d(h_conv4,6890))
                h_conv4=tf.reshape(h_conv4,[h_conv4.shape[0],1,h_conv4.shape[1],h_conv4.shape[2]])
#                print h_conv4
                h_up1 = upsample_layer(h_conv4, out_channels_conv4, "wen_deconv2", 2,N)#13141
                h_up1=tf.reshape(h_up1,[h_conv4.shape[0],N,out_channels_conv4])

#    #            adj_up0=un_max_pooling_1d(h_conv3,3445)
#                de_perm_up1=perm_tensor(h_up1,x_de_perm)
#                h_up1=de_perm_up1[:,:6890,:]
                print "h_up1:"
                print h_up1
            
            # Cont1  
            with tf.variable_scope('Cont1'):
                h_cont1 = tf.nn.relu(tf.concat([h_conv0,h_up1],2))
    #            adj_up0 = tf.cast(adj_up0, dtype=tf.int32)
    #            adj_cont0 = tf.concat([c,adj_up0],1)
                print "h_cont1:"
                print h_cont1
            
            # Lin(1024)
            with tf.variable_scope('Lin0'):
                out_channels_fc1 = 256 #1024
                h_fc1 = tf.nn.relu(custom_lin(h_cont1, out_channels_fc1))
                print h_fc1
            
            # Lin(num_classes)
            with tf.variable_scope('Lin1'):
                y_conv = custom_lin(h_fc1, num_classes)
                print y_conv
            
            print('lx debug', adj.shape)
            points_w, nbr_idc = get_lleWb(x,adj)
            lle_loss = tf.nn.l2_loss(tf.matmul(points_w,h_cont1)-tf.squeeze(h_cont1))/(B*num_classes*16) 
            
            return y_conv, lle_loss
        
# def get_model_lle(x, adj, num_classes, architecture, Mconv=9, reuse=False):
#     """ 
#     0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
#     """
# #    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
#     with tf.variable_scope('FeastNet'):
#     #     if reuse:                                        ### 改动部分 ###
#     #         vs.get_variable_scope().reuse_variables()
#         if architecture == 0:
#             points_w, nbr_idc = get_lleWb(x, adj)
#             B = x.get_shape()[0].value
#             with tf.variable_scope('FC0'):
#                 out_channels_fc0 = 16
#                 h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
#                 print h_fc0
#             # Conv1
#             with tf.variable_scope('Conv1'):
#                 M_conv1 = Mconv
#                 out_channels_conv1 = 32
#                 h_conv1 = tf.nn.relu(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1))
#                 print h_conv1
#             # Conv2
#             with tf.variable_scope('Conv2'):
#                 M_conv2 = Mconv
#                 out_channels_conv2 = 64
#                 h_conv2 = tf.nn.relu(custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2))
#                 print h_conv2
#             # Conv3
#             with tf.variable_scope('Conv3'):
#                 M_conv3 = Mconv
#                 out_channels_conv3 = 128
#                 h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3))
#                 print('h_conv3',h_conv3)
#             # Lin(1024)
#             with tf.variable_scope('Lin0'):
#                 out_channels_fc1 = 1024
#                 h_fc1 = tf.nn.relu(custom_lin(h_conv3, out_channels_fc1))
#                 print('h_fc1',h_fc1)
                
#             lle_loss = tf.nn.l2_loss(tf.matmul(points_w,h_conv3)-tf.squeeze(h_conv3))/(B*num_classes*256)
            
#             with tf.variable_scope('Lin1'):
#             # Lin(num_classes)
#                 y_conv = custom_lin(h_fc1, num_classes)
#                 print y_conv
#             return y_conv, lle_loss

def get_model_lle(x, adj, num_classes, architecture, Mconv=9, reuse=False):
    """ 
    0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
    """
#    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('FeastNet'):
    #     if reuse:                                        ### 改动部分 ###
    #         vs.get_variable_scope().reuse_variables()
        if architecture == 0:
            points_w, nbr_idc = get_lleWb(x, adj)
            B = x.get_shape()[0].value
            with tf.variable_scope('FC0'):
                out_channels_fc0 = 16
                h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
                print h_fc0
            # Conv1
            with tf.variable_scope('Conv1'):
                M_conv1 = Mconv
                out_channels_conv1 = 16
                h_conv1 = tf.nn.relu(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1))
                print h_conv1
            # Conv2
            with tf.variable_scope('Conv2'):
                M_conv2 = Mconv
                out_channels_conv2 = 16
                h_conv2 = tf.nn.relu(custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2))
                print h_conv2
            # Conv3
            with tf.variable_scope('Conv3'):
                M_conv3 = Mconv
                out_channels_conv3 = 16
                h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3))
                print('h_conv3',h_conv3)
            # Lin(1024)
            with tf.variable_scope('Lin0'):
                out_channels_fc1 = 256
                h_fc1 = tf.nn.relu(custom_lin(h_conv3, out_channels_fc1))
                print('h_fc1',h_fc1)
                
            lle_loss = tf.nn.l2_loss(tf.matmul(points_w,h_conv3)-tf.squeeze(h_conv3))/(B*num_classes*256)
            
            with tf.variable_scope('Lin1'):
            # Lin(num_classes)
                y_conv = custom_lin(h_fc1, num_classes)
                print y_conv
            return y_conv, lle_loss

# def get_model_lle(x, adj, num_classes, architecture, Mconv=9, reuse=False):
#     """ 
#     0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
#     """
# #    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
#     with tf.variable_scope('FeastNet'):
#     #     if reuse:                                        ### 改动部分 ###
#     #         vs.get_variable_scope().reuse_variables()
#         if architecture == 0:
#             points_w, nbr_idc = get_lleWb(x, adj)
#             B = x.get_shape()[0].value
#             with tf.variable_scope('FC0'):
#                 out_channels_fc0 = 16
#                 h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
#                 print h_fc0
#             # Conv1
#             with tf.variable_scope('Conv1'):
#                 M_conv1 = Mconv
#                 out_channels_conv1 = 32
#                 h_conv1 = tf.nn.relu(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1))
#                 print h_conv1
#             # Conv2
#             with tf.variable_scope('Conv2'):
#                 M_conv2 = Mconv
#                 out_channels_conv2 = 64
#                 h_conv2 = tf.nn.relu(custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2))
#                 print h_conv2
#             # Conv3
#             with tf.variable_scope('Conv3'):
#                 M_conv3 = Mconv
#                 out_channels_conv3 = 128
#                 h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3))
#                 print('h_conv3',h_conv3)
#             # Lin(1024)
#             with tf.variable_scope('Lin0'):
#                 out_channels_fc1 = 256 #1024
#                 h_fc1 = tf.nn.relu(custom_lin(h_conv3, out_channels_fc1))
#                 print('h_fc1',h_fc1)
                
#             lle_loss = tf.nn.l2_loss(tf.matmul(points_w,h_conv3)-tf.squeeze(h_conv3))/(B*num_classes*256)
            
#             with tf.variable_scope('Lin1'):
#             # Lin(num_classes)
#                 y_conv = custom_lin(h_fc1, num_classes)
#                 print y_conv
            # return y_conv, lle_loss#, h_fc0, h_fc1

# def get_model_lle_un(x, adj, num_classes, architecture, Mconv=9, reuse=False):
#     """ 
#     0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
#     """
# #    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
#     with tf.variable_scope('FeastNet'):
#     #     if reuse:                                        ### 改动部分 ###
#     #         vs.get_variable_scope().reuse_variables()
#         if architecture == 0:
#             points_w, nbr_idc = get_lleWb(x, K=6)
#             B = x.get_shape()[0].value
#             with tf.variable_scope('FC0'):
#                 out_channels_fc0 = 16
#                 h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
#                 print h_fc0
#             # Conv1
#             with tf.variable_scope('Conv1'):
#                 M_conv1 = Mconv
#                 out_channels_conv1 = 32
#                 h_conv1 = tf.nn.relu(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1))
#                 print h_conv1
#             # Conv2
#             with tf.variable_scope('Conv2'):
#                 M_conv2 = Mconv
#                 out_channels_conv2 = 64
#                 h_conv2 = tf.nn.relu(custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2))
#                 print h_conv2
#             # Conv3
#             with tf.variable_scope('Conv3'):
#                 M_conv3 = Mconv
#                 out_channels_conv3 = 128
#                 h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3))
#                 print h_conv3
#             # Lin(1024)
#             with tf.variable_scope('Lin0'):
#                 out_channels_fc1 = 256 #1024
#                 h_fc1 = tf.nn.relu(custom_lin(h_conv3, out_channels_fc1))
#                 print h_fc1
                
#             lle_loss = tf.nn.l2_loss(tf.matmul(points_w,tf.squeeze(h_conv3))-tf.squeeze(h_conv3))/(B*num_classes*256)
            
#             # with tf.variable_scope('Lin1'):
#             # # Lin(num_classes)
#             #     y_conv = custom_lin(h_fc1, num_classes)
#             #     print y_conv
#             return h_fc1,lle_loss

# def get_model_lle_un2(x, adj, num_classes, architecture, Mconv=9, reuse=False):
#     """ 
#     0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
#     """
# #    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
#     with tf.variable_scope('FeastNet'):
#     #     if reuse:                                        ### 改动部分 ###
#     #         vs.get_variable_scope().reuse_variables()
#         if architecture == 0:
#             points_w, nbr_idc = get_lleWb(x, K=6)
#             B = x.get_shape()[0].value
#             with tf.variable_scope('FC0'):
#                 out_channels_fc0 = 16
#                 h_fc0 = tf.nn.relu(custom_lin(x, out_channels_fc0))
#                 print h_fc0
#             # Conv1
#             with tf.variable_scope('Conv1'):
#                 M_conv1 = Mconv
#                 out_channels_conv1 = 32
#                 h_conv1 = tf.nn.relu(custom_conv2d(h_fc0, adj, out_channels_conv1, M_conv1))
#                 print h_conv1
#             # Conv2
#             with tf.variable_scope('Conv2'):
#                 M_conv2 = Mconv
#                 out_channels_conv2 = 64
#                 h_conv2 = tf.nn.relu(custom_conv2d(h_conv1, adj, out_channels_conv2, M_conv2))
#                 print h_conv2
#             # Conv3
#             with tf.variable_scope('Conv3'):
#                 M_conv3 = Mconv
#                 out_channels_conv3 = 128
#                 h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj, out_channels_conv3, M_conv3))
#                 print h_conv3
#             # Lin(1024)
#             with tf.variable_scope('Lin0'):
#                 out_channels_fc1 = 1024
#                 h_fc1 = tf.nn.relu(custom_lin(h_conv3, out_channels_fc1))
#                 print h_fc1
            
#             # MaxPooling(1024)
#             with tf.variable_scope('Maxpooling0'):
#                 h_fc1_c = tf.expand_dims(h_fc1, axis=2)
#                 gb = tf.nn.max_pool(h_fc1_c, ksize=[1, 6890, 1, 1], strides=[1, 6890, 1, 1], padding='VALID')
#                 print gb
#             gb = tf.tile(gb,[1,6890,1,1])
#             expand = tf.squeeze(gb, axis=2)
#             features = tf.concat([h_conv1, h_conv2, h_conv3, h_fc1, expand], axis=-1)
#             lle_loss = tf.nn.l2_loss(tf.matmul(points_w,tf.squeeze(h_conv3))-tf.squeeze(h_conv3))/(B*num_classes*256)
            
#             # with tf.variable_scope('Lin1'):
#             # # Lin(num_classes)
#             #     y_conv = custom_lin(h_fc1, num_classes)
#             #     print y_conv
#             return features,lle_loss


def get_model_multi_xyz(x, adj, num_classes, architecture, Mconv=9, is_training=True, reuse=False):
    """ 
    0 - input(N*3) - CONV(N*8) - POOL(N/2*8) - CONV(N/2*16) - POOL(N/4*16) - CONV(N/4*32) - CONV(N/4*16) - UP+CONCAT(N/2*32) - UP+CONCAT(N*16) - LIN(N*256)- Output(N*N)
    """
    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
    # with tf.variable_scope('FeastNet'):
#         if reuse:                                        ### 改动部分 ###
#             vs.get_variable_scope().reuse_variables()
        if architecture == 0:
            
            B = x.get_shape()[0].value
            N = x.get_shape()[1].value
            print x
            
            is_training = tf.cast(is_training, tf.bool)
            
            with tf.variable_scope('transform_net1') as sc:
                sc.reuse_variables()
                transform = input_transform_net(x,is_training)
                x_transformed = tf.matmul(x, transform)
            
            # Conv0
            with tf.variable_scope('FC0'):
                M_conv0 = Mconv
                out_channels_conv0 = 8
                h_conv0 = tf.nn.relu(custom_conv2d(x_transformed, adj,out_channels_conv0,M_conv0))
                print "h_conv0:"
                print h_conv0
            
            # Pool0
            with tf.variable_scope('Pool0'):
#            h_pool0 = tf.nn.relu(wen_max_pool(h_conv0))
#                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
#                h_pool0 = tf.layers.max_pooling1d(h_conv0,2,2,padding='same')
#                adj_pool = tf.layers.max_pooling1d(adj,2,2,padding='same')
#                adj_pool = tf.nn.relu(tf.layers.max_pooling1d(adj,2,2,padding='same'))
#                h_pool0 = tf.nn.relu(wen_max_pool(h_conv0,x_perm))
#                adj=tf.cast(adj, dtype=tf.float32)
#                adj_pool = wen_max_pool(adj,x_perm)
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                
#                h_pool0,adj_pool = wen_permute_pool(h_conv0,adj,None)
               
                
                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
     
                batch_size, point_size, channels = adj.get_shape().as_list()
                if point_size%2==1:
                    adj_t=tf.concat([adj,tf.zeros([batch_size,1,channels],dtype=tf.int32)],axis=1)
                    print('lx debug', adj_t.shape)
                    batch_size, point_size, channels = adj_t.get_shape().as_list()
                    adj_pool=tf.reshape(adj_t,[batch_size,int(point_size/2),int(channels*2)])
                else:
                    adj_pool=tf.reshape(adj,[batch_size,int(point_size/2),int(channels*2)])
                adj_pool1=(adj_pool+1)/2
                adj_pool1=tf.cast(adj_pool1, dtype=tf.int32)
    
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                
#                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
##                adj_pool= custom_pool(adj,tf.convert_to_tensor(range(0,6890,2)))
#                batch_size, point_size, channels = adj.get_shape().as_list()
#                adj_pool=tf.reshape(adj,[batch_size,int(point_size/2),int(channels*2)])
#                adj_pool=(adj_pool+1)/2
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                print("h_pool0:", h_pool0)
                print('adj_pool1', adj_pool1)
            
            # Conv1
            with tf.variable_scope('Conv1'):
                M_conv1 = Mconv
                out_channels_conv1 = 16
                h_conv1 = tf.nn.relu(custom_conv2d(h_pool0, adj_pool, out_channels_conv1, M_conv1))
                print "h_conv1:"
                print h_conv1
            
            # Pool1
            with tf.variable_scope('Pool1'):
    #            h_pool0 = tf.nn.relu(wen_max_pool(h_conv0))
#                h_pool1 = tf.nn.relu(tf.layers.max_pooling1d(h_conv1,2,2,padding='same'))
##                h_pool1 = tf.layers.max_pooling1d(h_conv1,2,2,padding='same')
#                adj_pool2 = tf.layers.max_pooling1d(adj_pool,2,2,padding='same')
#                adj_pool2 = tf.nn.relu(tf.layers.max_pooling1d(adj_pool,2,2,padding='same'))
    
#                h_pool1 = custom_pool(h_conv1,tf.convert_to_tensor(range(0,3446,2)))
#                adj_pool2= custom_pool(adj_pool,tf.convert_to_tensor(range(0,3446,2)))
                
                h_pool1 = tf.nn.relu(tf.layers.max_pooling1d(h_conv1,2,2,padding='same'))
#                adj_pool2= custom_pool(adj_pool,tf.convert_to_tensor(range(0,3446,2)))
                batch_size, point_size, channels = adj_pool1.get_shape().as_list()
                if point_size%2==1:
                    adj_t=tf.concat([adj_pool1,tf.zeros([batch_size,1,channels],dtype=tf.int32)],axis=1)
                    print('lx debug', adj_t.shape)
                    batch_size, point_size, channels = adj_t.get_shape().as_list()
                    adj_pool=tf.reshape(adj_t,[batch_size,int(point_size/2),int(channels*2)]) #(2,3446,16)->(2,1722,24)
                else:
                    adj_pool=tf.reshape(adj_pool1,[batch_size,int(point_size/2),int(channels*2)])
                adj_pool2=(adj_pool+1)/2
                adj_pool2=tf.cast(adj_pool, dtype=tf.int32)
                print "h_pool1:"
                print h_pool1
                print adj_pool2
            
            # Conv2
            with tf.variable_scope('Conv2'):
                M_conv2 = Mconv
                out_channels_conv2 = 32
                h_conv2 = tf.nn.relu(custom_conv2d(h_pool1, adj_pool2, out_channels_conv2, M_conv2))
                print "h_conv2:"
                print h_conv2
            
            # Conv3
            with tf.variable_scope('Conv3'):
                M_conv3 = Mconv
                out_channels_conv3 = 16
                h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj_pool2, out_channels_conv3, M_conv3))
                print "h_conv3:"
                print h_conv3
            
            # UP0
            with tf.variable_scope('UP0'):
                
#                h_up0 = tf.nn.relu(wen_un_pool(h_conv3,3445))
                
#                h_up0 = tf.nn.relu(un_max_pooling_1d(h_conv3,3446))
                h_conv3=tf.reshape(h_conv3,[h_conv3.shape[0],1,h_conv3.shape[1],h_conv3.shape[2]])
#                print h_conv3
#                print out_channels_conv3
                
#                h_up0 = tf.layers.conv2d_transpose(h_conv3, out_channels_conv3, 2, strides=[ 1, 2], padding='same')
#                print h_up0
                h_up0 = upsample_layer(h_conv3, out_channels_conv3, "wen_deconv", 2,int((N+1)/2))#6571
#                print h_up0
                h_up0=tf.reshape(h_up0,[h_conv3.shape[0],int((N+1)/2),out_channels_conv3])
    #            adj_up0=un_max_pooling_1d(h_conv3,3445)
                print "h_up0:"
                print h_up0
            
            # Cont0  
            with tf.variable_scope('Cont0'):
                h_cont0 = tf.nn.relu(tf.concat([h_conv1,h_up0],2))
#                h_cont0 = tf.nn.relu(tf.concat([h_up0,h_conv1],2))
    #            adj_up0 = tf.cast(adj_up0, dtype=tf.int32)
    #            adj_cont0 = tf.concat([c,adj_up0],1)
                print "h_cont0:"
                print h_cont0
            
             # Conv4
            with tf.variable_scope('Conv4'):
                M_conv4 = Mconv
                out_channels_conv4 = 8
                h_conv4 = tf.nn.relu(custom_conv2d(h_cont0, adj_pool1, out_channels_conv4, M_conv4))
                print "h_conv4:"
                print h_conv4
            
             # UP1
            with tf.variable_scope('UP1'):
#                h_up1 = tf.nn.relu(wen_un_pool(h_conv4,6890))
                
#                h_up1 = tf.nn.relu(un_max_pooling_1d(h_conv4,6890))
                h_conv4=tf.reshape(h_conv4,[h_conv4.shape[0],1,h_conv4.shape[1],h_conv4.shape[2]])
#                print h_conv4
                h_up1 = upsample_layer(h_conv4, out_channels_conv4, "wen_deconv2", 2,N)#13141
                h_up1=tf.reshape(h_up1,[h_conv4.shape[0],N,out_channels_conv4])

#    #            adj_up0=un_max_pooling_1d(h_conv3,3445)
#                de_perm_up1=perm_tensor(h_up1,x_de_perm)
#                h_up1=de_perm_up1[:,:6890,:]
                print "h_up1:"
                print h_up1
            
            # Cont1  
            with tf.variable_scope('Cont1'):
                h_cont1 = tf.nn.relu(tf.concat([h_conv0,h_up1],2))
    #            adj_up0 = tf.cast(adj_up0, dtype=tf.int32)
    #            adj_cont0 = tf.concat([c,adj_up0],1)
                print "h_cont1:"
                print h_cont1
            
            # Lin(1024)
            with tf.variable_scope('Lin0'):
                out_channels_fc1 = 256 #1024
                h_fc1 = tf.nn.relu(custom_lin(h_cont1, out_channels_fc1))
                print h_fc1
            
            with tf.variable_scope('Cont2'):
                h_cont2 = tf.nn.relu(tf.concat([h_fc1,x_transformed],2))
    #            adj_up0 = tf.cast(adj_up0, dtype=tf.int32)
    #            adj_cont0 = tf.concat([c,adj_up0],1)
                print "h_cont2:"
                print h_cont2
            
            # Lin(num_classes)
            with tf.variable_scope('Lin1'):
                y_conv = custom_lin(h_cont2, num_classes)
                print y_conv
            
            return y_conv
        
def get_model_multi_xyz_lle(x, adj, num_classes, architecture, Mconv=9, is_training=True, reuse=False):
    """ 
    0 - input(N*3) - CONV(N*8) - POOL(N/2*8) - CONV(N/2*16) - POOL(N/4*16) - CONV(N/4*32) - CONV(N/4*16) - UP+CONCAT(N/2*32) - UP+CONCAT(N*16) - LIN(N*256)- Output(N*N)
    """
    with tf.sg_context(name='FeastNet', reuse=tf.AUTO_REUSE):
    # with tf.variable_scope('FeastNet'):
#         if reuse:                                        ### 改动部分 ###
#             vs.get_variable_scope().reuse_variables()
        if architecture == 0:
            
            B = x.get_shape()[0].value
            N = x.get_shape()[1].value
            print x
            
            is_training = tf.cast(is_training, tf.bool)
            
            with tf.variable_scope('transform_net1') as sc:
                sc.reuse_variables()
                transform = input_transform_net(x,is_training)
                x_transformed = tf.matmul(x, transform)
            
            # Conv0
            with tf.variable_scope('FC0'):
                M_conv0 = Mconv
                out_channels_conv0 = 8
                h_conv0 = tf.nn.relu(custom_conv2d(x_transformed, adj,out_channels_conv0,M_conv0))
                print "h_conv0:"
                print h_conv0
            
            # Pool0
            with tf.variable_scope('Pool0'):
#            h_pool0 = tf.nn.relu(wen_max_pool(h_conv0))
#                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
#                h_pool0 = tf.layers.max_pooling1d(h_conv0,2,2,padding='same')
#                adj_pool = tf.layers.max_pooling1d(adj,2,2,padding='same')
#                adj_pool = tf.nn.relu(tf.layers.max_pooling1d(adj,2,2,padding='same'))
#                h_pool0 = tf.nn.relu(wen_max_pool(h_conv0,x_perm))
#                adj=tf.cast(adj, dtype=tf.float32)
#                adj_pool = wen_max_pool(adj,x_perm)
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                
#                h_pool0,adj_pool = wen_permute_pool(h_conv0,adj,None)
               
                
                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
     
                batch_size, point_size, channels = adj.get_shape().as_list()
                if point_size%2==1:
                    adj_t=tf.concat([adj,tf.zeros([batch_size,1,channels],dtype=tf.int32)],axis=1)
                    print('lx debug', adj_t.shape)
                    batch_size, point_size, channels = adj_t.get_shape().as_list()
                    adj_pool=tf.reshape(adj_t,[batch_size,int(point_size/2),int(channels*2)])
                else:
                    adj_pool=tf.reshape(adj,[batch_size,int(point_size/2),int(channels*2)])
                adj_pool1=(adj_pool+1)/2
                adj_pool1=tf.cast(adj_pool1, dtype=tf.int32)
    
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                
#                h_pool0 = tf.nn.relu(tf.layers.max_pooling1d(h_conv0,2,2,padding='same'))
##                adj_pool= custom_pool(adj,tf.convert_to_tensor(range(0,6890,2)))
#                batch_size, point_size, channels = adj.get_shape().as_list()
#                adj_pool=tf.reshape(adj,[batch_size,int(point_size/2),int(channels*2)])
#                adj_pool=(adj_pool+1)/2
#                adj_pool=tf.cast(adj_pool, dtype=tf.int32)
                print("h_pool0:", h_pool0)
                print('adj_pool1', adj_pool1)
            
            # Conv1
            with tf.variable_scope('Conv1'):
                M_conv1 = Mconv
                out_channels_conv1 = 16
                h_conv1 = tf.nn.relu(custom_conv2d(h_pool0, adj_pool, out_channels_conv1, M_conv1))
                print "h_conv1:"
                print h_conv1
            
            # Pool1
            with tf.variable_scope('Pool1'):
    #            h_pool0 = tf.nn.relu(wen_max_pool(h_conv0))
#                h_pool1 = tf.nn.relu(tf.layers.max_pooling1d(h_conv1,2,2,padding='same'))
##                h_pool1 = tf.layers.max_pooling1d(h_conv1,2,2,padding='same')
#                adj_pool2 = tf.layers.max_pooling1d(adj_pool,2,2,padding='same')
#                adj_pool2 = tf.nn.relu(tf.layers.max_pooling1d(adj_pool,2,2,padding='same'))
    
#                h_pool1 = custom_pool(h_conv1,tf.convert_to_tensor(range(0,3446,2)))
#                adj_pool2= custom_pool(adj_pool,tf.convert_to_tensor(range(0,3446,2)))
                
                h_pool1 = tf.nn.relu(tf.layers.max_pooling1d(h_conv1,2,2,padding='same'))
#                adj_pool2= custom_pool(adj_pool,tf.convert_to_tensor(range(0,3446,2)))
                batch_size, point_size, channels = adj_pool1.get_shape().as_list()
                if point_size%2==1:
                    adj_t=tf.concat([adj_pool1,tf.zeros([batch_size,1,channels],dtype=tf.int32)],axis=1)
                    print('lx debug', adj_t.shape)
                    batch_size, point_size, channels = adj_t.get_shape().as_list()
                    adj_pool=tf.reshape(adj_t,[batch_size,int(point_size/2),int(channels*2)]) #(2,3446,16)->(2,1722,24)
                else:
                    adj_pool=tf.reshape(adj_pool1,[batch_size,int(point_size/2),int(channels*2)])
                adj_pool2=(adj_pool+1)/2
                adj_pool2=tf.cast(adj_pool, dtype=tf.int32)
                print "h_pool1:"
                print h_pool1
                print adj_pool2
            
            # Conv2
            with tf.variable_scope('Conv2'):
                M_conv2 = Mconv
                out_channels_conv2 = 32
                h_conv2 = tf.nn.relu(custom_conv2d(h_pool1, adj_pool2, out_channels_conv2, M_conv2))
                print "h_conv2:"
                print h_conv2
            
            # Conv3
            with tf.variable_scope('Conv3'):
                M_conv3 = Mconv
                out_channels_conv3 = 16
                h_conv3 = tf.nn.relu(custom_conv2d(h_conv2, adj_pool2, out_channels_conv3, M_conv3))
                print "h_conv3:"
                print h_conv3
            
            # UP0
            with tf.variable_scope('UP0'):
                
#                h_up0 = tf.nn.relu(wen_un_pool(h_conv3,3445))
                
#                h_up0 = tf.nn.relu(un_max_pooling_1d(h_conv3,3446))
                h_conv3=tf.reshape(h_conv3,[h_conv3.shape[0],1,h_conv3.shape[1],h_conv3.shape[2]])
#                print h_conv3
#                print out_channels_conv3
                
#                h_up0 = tf.layers.conv2d_transpose(h_conv3, out_channels_conv3, 2, strides=[ 1, 2], padding='same')
#                print h_up0
                h_up0 = upsample_layer(h_conv3, out_channels_conv3, "wen_deconv", 2,int((N+1)/2))#6571
#                print h_up0
                h_up0=tf.reshape(h_up0,[h_conv3.shape[0],int((N+1)/2),out_channels_conv3])
    #            adj_up0=un_max_pooling_1d(h_conv3,3445)
                print "h_up0:"
                print h_up0
            
            # Cont0  
            with tf.variable_scope('Cont0'):
                h_cont0 = tf.nn.relu(tf.concat([h_conv1,h_up0],2))
#                h_cont0 = tf.nn.relu(tf.concat([h_up0,h_conv1],2))
    #            adj_up0 = tf.cast(adj_up0, dtype=tf.int32)
    #            adj_cont0 = tf.concat([c,adj_up0],1)
                print "h_cont0:"
                print h_cont0
            
             # Conv4
            with tf.variable_scope('Conv4'):
                M_conv4 = Mconv
                out_channels_conv4 = 8
                h_conv4 = tf.nn.relu(custom_conv2d(h_cont0, adj_pool1, out_channels_conv4, M_conv4))
                print "h_conv4:"
                print h_conv4
            
             # UP1
            with tf.variable_scope('UP1'):
#                h_up1 = tf.nn.relu(wen_un_pool(h_conv4,6890))
                
#                h_up1 = tf.nn.relu(un_max_pooling_1d(h_conv4,6890))
                h_conv4=tf.reshape(h_conv4,[h_conv4.shape[0],1,h_conv4.shape[1],h_conv4.shape[2]])
#                print h_conv4
                h_up1 = upsample_layer(h_conv4, out_channels_conv4, "wen_deconv2", 2,N)#13141
                h_up1=tf.reshape(h_up1,[h_conv4.shape[0],N,out_channels_conv4])

#    #            adj_up0=un_max_pooling_1d(h_conv3,3445)
#                de_perm_up1=perm_tensor(h_up1,x_de_perm)
#                h_up1=de_perm_up1[:,:6890,:]
                print "h_up1:"
                print h_up1
            
            # Cont1  
            with tf.variable_scope('Cont1'):
                h_cont1 = tf.nn.relu(tf.concat([h_conv0,h_up1],2))
    #            adj_up0 = tf.cast(adj_up0, dtype=tf.int32)
    #            adj_cont0 = tf.concat([c,adj_up0],1)
                print "h_cont1:"
                print h_cont1
            
            # Lin(1024)
            with tf.variable_scope('Lin0'):
                out_channels_fc1 = 256 #1024
                h_fc1 = tf.nn.relu(custom_lin(h_cont1, out_channels_fc1))
                print h_fc1
            
            with tf.variable_scope('Cont2'):
                h_cont2 = tf.nn.relu(tf.concat([h_fc1,x_transformed],2))
    #            adj_up0 = tf.cast(adj_up0, dtype=tf.int32)
    #            adj_cont0 = tf.concat([c,adj_up0],1)
                print "h_cont2:"
                print h_cont2
            
            # Lin(num_classes)
            with tf.variable_scope('Lin1'):
                y_conv = custom_lin(h_cont2, num_classes)
                print y_conv
            
            print('lx debug', adj.shape)
            points_w, nbr_idc = get_lleWb(x,adj)
            lle_loss = tf.nn.l2_loss(tf.matmul(points_w,h_cont1)-tf.squeeze(h_cont1))/(B*num_classes*256) 
            
            return y_conv, lle_loss