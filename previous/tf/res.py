import argparse
import tensorflow as tf

def ResNet(input, seq_len, num_outputs, args):

	## RESNET
	network = [input]

	## INPUT LAYER
	if len(network) - 1 == 0 and args.input_layer:
		with tf.variable_scope('in_layer'):
			network.append(tf.multiply(tf.layers.dense(network[-1], args.input_size, tf.nn.relu, 
				bias_initializer=tf.constant_initializer([0.1]*args.input_size)), 
				tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32))) # FC.
	
	## RESIDUAL BLOCKS
	for i in range(args.rnn_depth):
		with tf.variable_scope('B' + str(i + 1)):
			start_layer = len(network) - 1 # starting index of block.
					
			## LSTMP CELL
			if args.bidirectional:
				network.append(blstmp_layer(network[-1], args.cell_size, 
					seq_len, args.peepholes, args.cell_proj, args.parallel_iterations)) # BLSTMP.
			else:
				network.append(lstmp_layer(network[-1], args.cell_size, seq_len, 
					args.peepholes, args.cell_proj, args.parallel_iterations)) # LSTMP.
			
			## RESIDUAL CONNECTION (ADDITION)
			if args.residual == 'add':
				if network[-1].get_shape().as_list()[-1] == network[start_layer].get_shape().as_list()[-1]:
					with tf.variable_scope('add_L' + str(len(network)-1) + '_L' + str(start_layer)):
						network.append(tf.add(network[-1], network[start_layer])) # residual connection.
			
			## RESIDUAL CONNECTION (CONCATENATION)
			if args.residual == 'concat':
				with tf.variable_scope('concat_L' + str(len(network)-1) + '_L' + str(start_layer)):
					network.append(tf.concat([network[-1], network[start_layer]], 2))
				with tf.variable_scope('proj'):
					network.append(tf.multiply(tf.layers.dense(network[-1], args.res_proj, use_bias = False), 
						tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), 2), tf.float32))) # projection.

	## OUTPUT LAYER
	with tf.variable_scope('out_layer'):
		network.append(tf.boolean_mask(network[-1], tf.sequence_mask(seq_len)))
		network.append(tf.layers.dense(network[-1], num_outputs))

	## SUMMARY	
	if args.verbose:
		for I, i in enumerate(network):
			print(i.get_shape().as_list(), end="")
			print("%i:" % (I), end="")
			print(str(i.name))

	return network[-1]

## RNN LAYER WITH LSTMP CELLS
def lstmp_layer(input, cell_dim, seq_len, peepholes, cell_proj, par_iter):
	with tf.variable_scope('LSTMP_layer'):
		cell = tf.contrib.rnn.LSTMCell(cell_dim, use_peepholes=False, num_proj=cell_proj) # BasicLSTMCell.
		output, _ = tf.nn.dynamic_rnn(cell, input, seq_len, swap_memory=True, parallel_iterations=par_iter, 
			dtype=tf.float32) # Recurrent Neural Network.
		return output

## BRNN LAYER WITH LSTMP CELLS
def blstmp_layer(input, cell_dim, seq_len, peepholes, cell_proj, par_iter):
	with tf.variable_scope('BLSTMP_layer'):
		cell_fw = tf.contrib.rnn.LSTMCell(cell_dim, use_peepholes=False, num_proj = cell_proj) # forward BasicLSTMCell.
		cell_bw = tf.contrib.rnn.LSTMCell(cell_dim, use_peepholes=False, num_proj = cell_proj) # backward BasicLSTMCell.
		output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input, seq_len, 
			swap_memory=True, parallel_iterations=par_iter, dtype=tf.float32) # Bidirectional Recurrent Neural Network.
		return tf.concat(output, 2) # concatenate forward and backward outputs.

## LOSS FUNCTIONS
def loss(target, estimate, loss_fnc):
  'loss functions for gradient descent.'
  with tf.name_scope(loss_fnc + '_loss'):
    if loss_fnc == 'mse':
      loss = tf.losses.mean_squared_error(labels=target, predictions=estimate)
    if loss_fnc == 'softmax_xentropy':
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=estimate)
    if loss_fnc == 'sigmoid_xentropy':
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=estimate)
    return loss

## GRADIENT DESCENT OPTIMISERS
def optimizer(loss, lr=None, epsilon=None, var_list=None, optimizer='adam'):
  'optimizers for training.'
  with tf.name_scope(optimizer + '_opt'):
    if optimizer == 'adam':
      if lr == None: lr = 0.001 # default.
      if epsilon == None: epsilon = 1e-8 # default.
      optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
      trainer = optimizer.minimize(loss, var_list=var_list) 
    if optimizer == 'nadam':
      if lr == None: lr = 0.001 # default.
      if epsilon == None: epsilon = 1e-8 # default.
      optimizer =  tf.contrib.opt.NadamOptimizer(learning_rate=lr, epsilon=epsilon)
      trainer = optimizer.minimize(loss, var_list=var_list) 
    if optimizer == 'sgd':
      if lr == None:
        lr = 0.5 # default.
      optimizer = tf.train.GradientDescentOptimizer(lr)
      trainer = optimizer.minimize(loss, var_list=var_list) 
    return trainer, optimizer

