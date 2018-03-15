import tensorflow as tf
import operations as op
import numpy as np

## FILE:           create.py 
## DATE:           2017
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Creates TensorFlow graph layers.

######################
## CREATE RNN LAYER ##
######################
class rnn_layer:
	def __init__(self, input, seq_len, cell_dim, layer, **opts):
		with tf.name_scope('rnn_layer_' + layer):

			## I/O ##
			self.layer = layer # layer number.
			self.input = input # layer input.
			self.seq_len = seq_len # sequence length placeholder.
			self.cell_dim = cell_dim # cell dimensions.
			self.output_dim = self.cell_dim # number of output dimensions.
			self.opts = opts # options for the layer.

			## GPU OPTIONS ##
			self.opts.setdefault('swap_memory', False) # swap memory.
			self.opts.setdefault('parallel_iterations', 32) # number of parallel iterations.
			
			## RNN OPTIONS ##
			self.opts.setdefault('cell', 'BasicLSTMCell') # cell type.
			self.opts.setdefault('peepholes', False) # peepholes.
			self.opts.setdefault('keep_prob', 1.0) # dropout keep probability.
			self.opts.setdefault('feature_size', None) # feature size.
			self.opts.setdefault('frequency_skip', None) # frequency skip.

			## RNN ##
			with tf.name_scope(self.opts['cell']):
				if self.opts['cell'] is ('BasicRNNCell'):
					self.cell = tf.contrib.rnn.BasicRNNCell(self.cell_dim) # BasicRNNCell.
				elif self.opts['cell'] is ('BasicLSTMCell'):
					self.cell = tf.contrib.rnn.BasicLSTMCell(self.cell_dim) # BasicLSTMCell.
				elif self.opts['cell'] is ('LSTMCell'):
					self.cell = tf.contrib.rnn.LSTMCell(self.cell_dim, use_peepholes=self.opts['peepholes']) # LSTMCell.
				elif self.opts['cell'] is ('LayerNormBasicLSTMCell'):	
					self.cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.cell_dim, dropout_keep_prob=self.opts['keep_prob']) # LayerNormBasicLSTMCell. 
				elif self.opts['cell'] is ('GRUCell'):	
					self.cell = tf.contrib.rnn.GRUCell(self.cell_dim) # GRUCell.

				## DYNAMIC RNN ##
				self.output, self.state = tf.nn.dynamic_rnn(self.cell, self.input, self.seq_len, swap_memory=self.opts['swap_memory'],
					parallel_iterations=self.opts['parallel_iterations'], dtype=tf.float32, scope=('rnn_layer_' + layer))
	
####################################
## CREATE BIDIRECTIONAL RNN LAYER ##
####################################
class brnn_layer:
	def __init__(self, input, seq_len, cell_dim, layer, **opts):
		with tf.name_scope('brnn_layer_' + layer):

			## I/O ##
			self.layer = layer # layer number.
			self.input = input # layer input.
			self.seq_len = seq_len # sequence length placeholder.
			self.cell_dim = cell_dim # cell dimensions.
			self.output_dim = int(np.multiply(self.cell_dim, 2)) # number of output dimensions.
			self.opts = opts # options for the layer.

			## GPU OPTIONS ##
			self.opts.setdefault('swap_memory', False) # swap memory.
			self.opts.setdefault('parallel_iterations', 32) # number of parallel iterations.
			
			## RNN OPTIONS ##
			self.opts.setdefault('cell', 'BasicLSTMCell') # cell type.
			self.opts.setdefault('peepholes', False) # peepholes.
				
			## BIDIRECTIONAL RNN ##
			with tf.name_scope(self.opts['cell']):
				if self.opts['cell'] is ('BasicRNNCell'):
					self.cell_fw = tf.contrib.rnn.BasicRNNCell(self.cell_dim) # forward BasicRNNCell.
					self.cell_bw = tf.contrib.rnn.BasicRNNCell(self.cell_dim) # backward BasicRNNCell.
				elif self.opts['cell'] is ('BasicLSTMCell'):
					self.cell_fw = tf.contrib.rnn.BasicLSTMCell(self.cell_dim) # forward BasicLSTMCell.
					self.cell_bw = tf.contrib.rnn.BasicLSTMCell(self.cell_dim) # backward BasicLSTMCell.
				elif self.opts['cell'] is ('LSTMCell'):
					self.cell_fw = tf.contrib.rnn.LSTMCell(self.cell_dim, use_peepholes=self.opts['peepholes']) # forward LSTMCell.
					self.cell_bw = tf.contrib.rnn.LSTMCell(self.cell_dim, use_peepholes=self.opts['peepholes']) # backward LSTMCell.
				elif self.opts['cell'] is ('GRUCell'):	
					self.cell_fw = tf.contrib.rnn.GRUCell(self.cell_dim) # forward GRUCell.
					self.cell_bw = tf.contrib.rnn.GRUCell(self.cell_dim) # backward GRUCell.

				## BIDIRECTIONAL DYNAMIC RNN ##
				self.output, self.state, = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, self.input, self.seq_len, 
					swap_memory=self.opts['swap_memory'], parallel_iterations=self.opts['parallel_iterations'], 
					dtype=tf.float32, scope=('brnn_layer_' + layer))
				self.output = tf.concat(self.output, 2) # concatenate forward and backward output.
		
##################################
## CREATE FULLY CONNECTED LAYER ##
##################################
class fc_layer:
	def __init__(self, input, num_inputs, num_outputs, layer, **opts):
		with tf.name_scope('fc_layer_' + layer):

			## I/O ##
			self.layer = layer # layer number.
			self.num_inputs = num_inputs # number of input dimensions.
			self.num_outputs  = num_outputs # number of output dimensions.
			self.input = input # layer input.
			self.opts = opts # options for the layer.

			## TENSORBOARD OPTIONS
			self.opts.setdefault('summaries', True) # tensorboard summaries.

			## ACTIVATION OPTIONS
			self.opts.setdefault('activation', 'sigmoid') # activation function.
			
			## DROPOUT OPTIONS
			self.opts.setdefault('dropout', False) # dropout applied to input of layer.
			self.opts.setdefault('keep_prob', None) # dropout data keeping probability.

			## OUTPUT LAYER PRE-TRAINING OPTIONS
			self.opts.setdefault('output_init', False) # output layer initialization.
			self.opts.setdefault('y', None) # targets.

			## VARIABLE INITIALIZATION OPTIONS    
			self.opts.setdefault('mean', 0.0) # mean.
			if self.opts['activation'] is ('relu','softplus'):
				self.opts.setdefault('stddev', np.sqrt(2/num_inputs)) # standard deviation.
			elif self.opts['activation'] == 'tanh':
				self.opts.setdefault('stddev', np.sqrt(2/(num_inputs+num_outputs))) # standard deviation.
			else:
				self.opts.setdefault('stddev', np.sqrt(1/num_inputs)) # standard deviation.

			## BIAS INITIALIZATION OPTIONS    
			self.opts.setdefault('constant', 0.1) # constant value for bias.

			#############
			## DROPOUT ##
			#############
			if self.opts['dropout']:
				self.input = tf.nn.dropout(self.input, self.opts['keep_prob']) # dropout applied to input

			#############
	    	## WEIGHTS ##
	    	#############
			self.weights = op.variable_init([self.num_inputs, self.num_outputs], self.opts['mean'], 
				self.opts['stddev'], self.opts['summaries'], 'weights') # weights variable.

			###########################
			## AFFINE TRANSFORMATION ##
			###########################
			self.bias = op.bias_init([self.num_outputs], self.opts['activation'], 
				self.opts['constant'], self.opts['mean'], self.opts['stddev'], self.opts['summaries'], 'bias') # bias variable.
			self.output = op.transform(self.input, self.weights, self.bias, 'affine') # apply transformation.
	      
			################
			## ACTIVATION ##
			################
			self.output = op.activation(self.output, self.opts['activation'], self.opts['summaries']) # apply activation function.

