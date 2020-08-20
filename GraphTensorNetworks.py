#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:59:11 2020

@author: yaoleixu
"""

# ============================================================================
# Import libraries
import numpy as np
import tensorflow as tf

tf.random.set_seed(0)


# ============================================================================
# General Multi-Graph Tensor Network Model
class MultiGraphTensorNetwork (tf.keras.layers.Layer):
    
    # Initialize terms
    def __init__(self, out_features_list, graphs_list, bias_bool=True, **kwargs):
        
        '''
        ------
        Inputs
        ------
        out_features_list: (list) list of feature maps per graph
        graphs_list: (list) list of graph adjacency matrices
        bias_bool: (bool) if use bias vector or not
        
        '''

        # Save input variables
        self.out_features_list = out_features_list
        self.graphs_list = graphs_list.copy()
        self.bias_bool = bias_bool
        
        # Useful variables
        self.n_graphs = len(graphs_list)
        self.graphs_shapes = [g.shape[0] for g in graphs_list]
        
        super(MultiGraphTensorNetwork, self).__init__(**kwargs)

    # Define weights
    def build(self, input_shape):
        
        # Create a list of kernels
        self.kernels = []
        self.r_kernels = []
        self.out_features_list = [input_shape[-1]] + self.out_features_list
        for i in range(self.n_graphs):
            
            # Feature map kernels
            self.kernels.append(self.add_weight(name='kernel{}'.format(i),
                                                shape=(self.out_features_list[i], self.out_features_list[i+1]),
                                                initializer='random_normal',
                                                trainable=True))
            
            # Graph propagation kernels
            self.r_kernels.append(self.add_weight(name='r_kernel{}'.format(i),
                                                 shape=(self.out_features_list[i+1], self.out_features_list[i+1]),
                                                 initializer='random_normal',
                                                 trainable=True))

            # Graph adjacency matrix casting
            self.graphs_list[i] = tf.constant(tf.cast(self.graphs_list[i], tf.float32))
                                        
        # Bias tensor
        if self.bias_bool:
            self.bias = self.add_weight(name='bias',
                                        shape=self.graphs_shapes + [self.out_features_list[-1]],
                                        initializer='uniform',
                                        trainable=True)
            
        # Be sure to call this at the end
        super(MultiGraphTensorNetwork, self).build(input_shape)

    # Forward pass
    def call(self, x):
                
        for i in range(self.n_graphs):
        
            # Compute the graph filter from the Kronecker product
            mat1 = tf.linalg.LinearOperatorFullMatrix(self.graphs_list[i]) # adjacency matrix (left)
            mat2 = tf.linalg.LinearOperatorFullMatrix(self.r_kernels[i]) # propagation matrix (right)
            kgf = tf.linalg.LinearOperatorKronecker([mat1, mat2]) # kronecker product
        
            # Generate feature map
            features = tf.tensordot(x, self.kernels[i], axes=[[-1],[0]])
        
            # Transpose and reshape to match kronecker filter
            transposing_axis = [m for m in range(self.n_graphs+1) if (m!=i+1)]+[i+1, self.n_graphs+1]
            features = tf.transpose(features, transposing_axis)
            org_shape = features.shape
            features = tf.reshape(features, [-1, self.graphs_shapes[i]*self.out_features_list[i+1]])
            features = tf.transpose(features, [1,0])
        
            # Filtering operation
            output = kgf.matmul(features) + features
        
            # Transposesand reshape back
            output = tf.transpose(output, [1,0])
            output = tf.reshape(output, [-1]+list(org_shape[1:]))
            output = tf.transpose(output, transposing_axis)
            
            x = output
        
        # Bias tensor
        if self.bias_bool: output = output + self.bias # add bias
                    
        return output

    # Compute output shape
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_features)
        
        
# ============================================================================
# Simple Multi-Graph Tensor Network Model
class SimpleMultiGraphTensorNetwork (tf.keras.layers.Layer):
    
    # Initialize terms
    def __init__(self, out_features, graphs_list, bias_bool=True, **kwargs):
        
        '''
        ------
        Inputs
        ------
        out_features: (int) number of feature maps
        graphs_list: (list) list of graph adjacency matrices
        bias_bool: (bool) if use bias vector or not
        
        '''

        # Save input variables
        self.out_features = out_features
        self.graphs_list = graphs_list.copy()
        self.bias_bool = bias_bool
        
        # Useful variables
        self.n_graphs = len(graphs_list)
        self.graphs_shapes = [g.shape[0] for g in graphs_list]
        
        super(SimpleMultiGraphTensorNetwork, self).__init__(**kwargs)

    # Define weights
    def build(self, input_shape):
        
        # Graph adjacency matrices: I+D^{0.5}AD^{0.5}
        for i in range(self.n_graphs):
            self.graphs_list[i] = tf.constant(tf.cast(self.graphs_list[i]+np.eye(self.graphs_shapes[i]), tf.float32))

        # Feature map kernel
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.out_features),
                                      initializer='uniform',
                                      trainable=True)
                                        
        # Bias tensor
        if self.bias_bool:
            self.bias = self.add_weight(name='bias',
                                        shape=self.graphs_shapes + [self.out_features],
                                        initializer='uniform',
                                        trainable=True)
                        
        # Be sure to call this at the end
        super(SimpleMultiGraphTensorNetwork, self).build(input_shape)

    # Forward pass
    def call(self, x):
                
        # Generate feature map
        output = tf.tensordot(x, self.kernel, axes=[[-1],[0]])

        # Graph filters
        for i in range(self.n_graphs):
            output = tf.tensordot(self.graphs_list[i], output, axes=[[-1],[i+1]])
        
        # Transpose back
        transposing_list = [i for i in range(self.n_graphs, -1, -1)] + [self.n_graphs+1]
        output = tf.transpose(output, transposing_list)
        
        # Bias tensor
        if self.bias_bool: output = output + self.bias # add bias
                    
        return output

    # Compute output shape
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_features)
        
        
# ============================================================================
# Special Multi Modal recurrent tensor network class
class SpecialMultiModalRecurrentTensorNetwork (tf.keras.layers.Layer):
    
    # Initialize terms
    def __init__(self, modes, out_features_list, tau, damping_constant=0.5, bias_bool=True, **kwargs):
        
        '''
        ------
        Inputs
        ------
        modes: (int) number of modes containing features
        out_features_list: (list) list of number of feature maps per feature mode
        tau: (int) number of time steps in consideration
        damping_constant: (float) damping constant for the time-shift
        bias_bool: (bool) if use bias vector or not
        
        '''

        # Save variables
        self.modes = modes
        self.out_features_list = out_features_list
        self.tau = tau
        self.damping_constant = damping_constant
        self.bias_bool = bias_bool
        
        super(SpecialMultiModalRecurrentTensorNetwork, self).__init__(**kwargs)

    # Define weights
    def build(self, input_shape):
        
        # Create a trainable weight variable for this layer of shape (f_i x f_o)
        self.kernels = []
        for m in range(self.modes):
            self.kernels.append(self.add_weight(name='kernel_{}'.format(m),
                                                shape=(input_shape[-self.modes+m], self.out_features_list[m]),
                                                initializer='uniform',
                                                trainable=True))
        
        # Adjacency matrix as a constant variable (N x N)
        A = np.zeros((self.tau, self.tau))
        for row_idx in range(A.shape[0]):
            for col_idx in range(A.shape[1]):
                if col_idx > row_idx:
                    A[row_idx, col_idx] = self.damping_constant**(col_idx-row_idx)

        self.A = tf.constant(tf.cast(A.T, tf.float32))
        self.G = self.A + tf.eye(self.tau)
                
        # Bias term of shape (F_o)
        if self.bias_bool:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.tau, )+tuple(self.out_features_list),
                                        initializer='uniform',
                                        trainable=True)

        # Be sure to call this at the end
        super(SpecialMultiModalRecurrentTensorNetwork, self).build(input_shape)

    # Forward pass
    def call(self, x):
        
        # Contractions for features generation
        features = x
        for m in range(self.modes):
            features = tf.tensordot(features, self.kernels[m], axes=[[-self.modes],[0]])
                                
        # graph filtering
        output = tf.tensordot(self.G, features, axes=[[1],[1]])
        
        # transpose into the right shape
        output = tf.transpose(output, perm=[1,0]+[i for i in range(2, 2+self.modes)])
        
        # Bias vector
        if self.bias_bool: output = output + self.bias
            
        return output

    # Compute output shape
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_features)
    

# ============================================================================
# General recurrent tensor network class
class GeneralRecurrentTensorNetwork (tf.keras.layers.Layer):
    
    # Initialize terms
    def __init__(self, out_features, tau, damping_constant=0.5, bias_bool=True, **kwargs):
        
        '''
        ------
        Inputs
        ------
        out_features: (int) number of feature maps
        tau: (int) number of time steps in consideration
        damping_constant: (float) damping constant for the time-shift
        bias_bool: (bool) if use bias vector or not
        
        '''

        # Save variables
        self.out_features = out_features
        self.tau = tau
        self.damping_constant = damping_constant
        self.bias_bool = bias_bool
        
        super(GeneralRecurrentTensorNetwork, self).__init__(**kwargs)

    # Define weights
    def build(self, input_shape):
        
        # Create a trainable weight variable for this layer of shape (f_i x f_o)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.out_features),
                                      initializer='uniform',
                                      trainable=True)
        
        # Recurrent kernel weights
        self.r_kernel = self.add_weight(name='r_kernel',
                                        shape=(self.out_features, self.out_features),
                                        initializer='uniform',
                                        trainable=True)

        # Adjacency matrix as a constant variable (N x N)
        A = np.zeros((self.tau, self.tau))
        for row_idx in range(A.shape[0]):
            for col_idx in range(A.shape[1]):
                if col_idx > row_idx:
                    A[row_idx, col_idx] = self.damping_constant**(col_idx-row_idx)
        self.A = tf.constant(tf.cast(A.T, tf.float32))
                
        # Bias term of shape (F_o)
        if self.bias_bool:
            self.bias = self.add_weight(name='bias',
                                        shape=(input_shape[-2], self.out_features, ),
                                        initializer='uniform',
                                        trainable=True)

        # Be sure to call this at the end
        super(GeneralRecurrentTensorNetwork, self).build(input_shape)

    # Forward pass
    def call(self, x):
        
        # Graph filter from the Kronecker product
        mat1 = tf.linalg.LinearOperatorFullMatrix(self.A)
        mat2 = tf.linalg.LinearOperatorFullMatrix(self.r_kernel)
        kgf = tf.linalg.LinearOperatorKronecker([mat1, mat2])
        
        # Feature map
        features = tf.tensordot(x, self.kernel, axes=[[2],[0]])
        
        # Graph filtering
        features = tf.reshape(features, [-1, self.tau*self.out_features])
        features = tf.transpose(features, [1,0])
        output = kgf.matmul(features) + features
        output = tf.transpose(output, [1,0])
        output = tf.reshape(output, [-1, self.tau, self.out_features])
        
        # Bias vector
        if self.bias_bool: output = output + self.bias # add bias
            
        return output

    # Compute output shape
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_features)


# ============================================================================
# Special recurrent tensor network class
class SpecialRecurrentTensorNetwork (tf.keras.layers.Layer):
    
    # Initialize terms
    def __init__(self, out_features, tau, damping_constant=0.5, bias_bool=True, **kwargs):
        
        '''
        ------
        Inputs
        ------
        out_features: (int) number of feature maps
        tau: (int) number of time steps in consideration
        damping_constant: (float) damping constant for the time-shift
        bias_bool: (bool) if use bias vector or not
        
        '''

        # Save variables
        self.out_features = out_features
        self.tau = tau
        self.damping_constant = damping_constant
        self.bias_bool = bias_bool
        
        super(SpecialRecurrentTensorNetwork, self).__init__(**kwargs)

    # Define weights
    def build(self, input_shape):
        
        # Create a trainable weight variable for this layer of shape (f_i x f_o)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.out_features),
                                      initializer='uniform',
                                      trainable=True)
        
        # Adjacency matrix as a constant variable (N x N)
        A = np.zeros((self.tau, self.tau))
        for row_idx in range(A.shape[0]):
            for col_idx in range(A.shape[1]):
                if col_idx > row_idx:
                    A[row_idx, col_idx] = self.damping_constant**(col_idx-row_idx)

        self.A = tf.constant(tf.cast(A.T, tf.float32))
        self.G = self.A + tf.eye(self.tau)
                
        # Bias term of shape (F_o)
        if self.bias_bool:
            self.bias = self.add_weight(name='bias',
                                        shape=(input_shape[-2], self.out_features, ),
                                        initializer='uniform',
                                        trainable=True)

        # Be sure to call this at the end
        super(SpecialRecurrentTensorNetwork, self).build(input_shape)

    # Forward pass
    def call(self, x):
        
        # Contractions
        features = tf.tensordot(x, self.kernel, axes=[[2],[0]]) # feature map
        output = tf.tensordot(self.G, features, axes=[[1],[1]]) # graph convolution
        output = tf.transpose(output, perm=[1,0,2]) # transpose into the right shape
        
        # Bias vector
        if self.bias_bool: output = output + self.bias # add bias
            
        return output

    # Compute output shape
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_features)
    
    
# ============================================================================
# Tensor-Train Fully-Connected Layer from Tensorizing Neural Networks
class TensorTrainLayer (tf.keras.layers.Layer):
    
    # define initial variables needed for implementation
    def __init__(self, tt_ips, tt_ops, tt_ranks, bias_bool=True, **kwargs):

        # Tensor Train Variables.
        self.tt_ips = np.array(tt_ips)
        self.tt_ops = np.array(tt_ops)
        self.tt_ranks = np.array(tt_ranks)
        self.num_dim = np.array(tt_ips).shape[0]
        self.param_n = np.sum(self.tt_ips*self.tt_ops*self.tt_ranks[1:]*self.tt_ranks[:-1])
        self.bias_bool = bias_bool
  
        super(TensorTrainLayer, self).__init__(**kwargs)

    # define weights for each core
    def build(self, input_shape):

        # Initalize weights for the TT FCL. Note that Keras will pass the optimizer directly on these core parameters
        self.cores = []
        for d in range(self.num_dim):
            if d == 0: my_shape = (self.tt_ips[d], self.tt_ops[d], self.tt_ranks[d+1])
            elif d == self.num_dim-1: my_shape = (self.tt_ranks[d], self.tt_ips[d], self.tt_ops[d])
            else: my_shape = (self.tt_ranks[d], self.tt_ips[d], self.tt_ops[d], self.tt_ranks[d+1])
            
            self.cores.append(self.add_weight(name='tt_core_{}'.format(d),
                                              shape=my_shape,
                                              initializer='uniform',
                                              trainable=True))
        
        # Bias vector
        if self.bias_bool:
            self.bias = self.add_weight(name='bias',
                                        shape=self.tt_ops,
                                        initializer='uniform',
                                        trainable=True)
        # Be sure to call this at the end
        super(TensorTrainLayer, self).build(input_shape)

    # Implementing the layer logic
    def call(self, x, mask=None):

        w = self.cores[0]
        for d in range(1, self.num_dim):
            w = tf.tensordot(w, self.cores[d], [[-1],[0]])

        output = tf.tensordot(x, w, [[i for i in range(1, 3+1)], [i for i in range(0, 2*3, 2)]])
        
        if self.bias_bool: output = output + self.bias

        return output

    # Compute input/output shapes
    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(self.tt_ops))
