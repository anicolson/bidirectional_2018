## FILE:           operations.py
## DATE:           2018
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
## BRIEF:          Operations for a TensorFlow graph.

import tensorflow as tf

def optimizer(loss, lr=None, epsilon=None, var_list=None, optimizer='adam'):
  'optimizers for training.'
  with tf.name_scope(optimizer + '_opt'):
    if optimizer == 'adam':
      if lr == None: lr = 0.001 # default.
      if epsilon == None: epsilon = 1e-8 # default.
      optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
      trainer = optimizer.minimize(loss, var_list=var_list) 
    if optimizer == 'sgd':
      if lr == None:
        lr = 0.5 # default.
      optimizer = tf.train.GradientDescentOptimizer(lr)
      trainer = optimizer.minimize(loss, var_list=var_list) 
    return trainer, optimizer

def accuracy(target, estimate):
  'accuracy for evaluation.'
  with tf.name_scope('acc'):
    correct_prediction = tf.equal(tf.argmax(estimate,1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
  return accuracy

def error(target, estimate, error_fnc):
  'error functions for evaluation.'
  with tf.name_scope(error_fnc + '_error'):
    if error_fnc == 'mse':
      error = tf.losses.mean_squared_error(target, estimate)
    if error_fnc == 'sigmoid_xentropy':
      error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=estimate))
    tf.summary.scalar((error_fnc + '_error'), error)
  return error

def loss(target, estimate, loss_fnc):
  'loss functions for gradient descent.'
  with tf.name_scope(loss_fnc + '_loss'):
    if loss_fnc == 'mse':
      loss = tf.losses.mean_squared_error(target, estimate)
    if loss_fnc == 'softmax_xentropy':
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=estimate))
    if loss_fnc == 'sigmoid_xentropy':
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=estimate))
    tf.summary.scalar((loss_fnc + '_loss'), loss)
    return loss

def transform(input, weights, bias=None, transform='affine'):
  'linear and affine transformations.'
  with tf.name_scope(transform + '_transform'):
    if transform == 'affine':
      output =  tf.add(tf.matmul(input, weights), bias)
    if transform == 'linear':
      output = tf.matmul(input, weights)
    return output

def activation(input, activation, summaries=False):
  'activation functions.'
  if activation in ('linear','affine'):
    return input
  with tf.name_scope(activation + '_function'): 
    if activation == 'sigmoid':
        output = tf.sigmoid(input)
    elif activation == 'relu':
        output = tf.nn.relu(input)
    elif activation == 'tanh':
        output = tf.tanh(input)
    elif activation == 'relu6':
        output = tf.nn.relu6(input)
    elif activation == 'crelu':
        output = tf.nn.crelu(input)
    elif activation == 'elu':
        output = tf.nn.elu(input)
    elif activation == 'softplus':
        output = tf.nn.softplus(input)
    elif activation == 'softsign':
        output = tf.nn.softsign(input)
    if summaries:
      variable_summaries(output)
    return output

def variable_init(shape, mean, stddev, summaries=False, name='Var'): 
  'initializes the variable randomly from a truncated normal distribution.'
  with tf.name_scope(name):
    var = tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=stddev), name=name) 
    if summaries:
      variable_summaries(var) # tensorboard summaries for the variable.
    return var

def bias_init(shape, activation='sigmoid', constant=0.1, mean=0.0, stddev=1.0, summaries=False, name='Var'): 
  'initializes the bias using a truncated normal distribution, or to a constant.'
  with tf.name_scope(name):
    if activation == 'relu':
      variable = tf.Variable(tf.constant(constant, shape=shape), name=name)
    else:
      variable = tf.Variable(tf.constant(0.0, shape=shape), name=name)
    if summaries:
      variable_summaries(variable) # tensorboard summaries for the variable.
    return variable

def reshape(input, shape):
  'reshape input.'
  with tf.name_scope('reshape'):
    return tf.reshape(input, shape)

def variable_summaries(variable):
  'tensorboard summaries for a variable.'
  with tf.name_scope('summaries'):
    tf.summary.scalar('mean', tf.reduce_mean(variable))
    tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(variable, tf.reduce_mean(variable))))))
    tf.summary.scalar('max', tf.reduce_max(variable))
    tf.summary.scalar('min', tf.reduce_min(variable))
    tf.summary.histogram('hist', variable)
    
