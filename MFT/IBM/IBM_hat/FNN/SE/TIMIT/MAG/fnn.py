# FILE:           fnn.py
# DATE:           2018
# AUTHOR:         Aaron Nicolson
# AFFILIATION:    Signal Processing Laboratory, Griffith University
# BRIEF:          Estimates the MS-IBM mask using a Feedforward Neural Network (FNN).

import tensorflow as tf
from tensorflow.python.data import Dataset, Iterator
import numpy as np
import feat, random, create, os, sets
import operations as op
from datetime import datetime
import scipy.io as spio
import time, sys
np.set_printoptions(threshold=np.nan)

'''
_np = numpy array,
_ph = placeholder, 
_d = dataset,
_i = iterator,
_g = mini-batch generator.
_m = mini-batch.
'''

## OPTIONS
training = False # perform training flag.
test = True # perform test flag.
save = True # save outputs.
version = 'FNN-1024-5' # model version.

## DATASETS
subset_path = '/home/aaron/datasets/tf_timit_se' # path to subsets.
noise_path = '/home/aaron/datasets/RSG-10/16KHz' # path to noise dataset.
save_path = '/home/aaron/feat/MFT/IBM/IBM_hat/FNN/SE/TIMIT/MAG/' + version # save path for output test files.
model_path = '/home/aaron/model/MFT/IBM/IBM_hat/FNN/SE/TIMIT/MAG/' + version # model save path.
if not os.path.exists(save_path): os.makedirs(save_path) # make save path directory.
if not os.path.exists(model_path): os.makedirs(model_path) # make model path directory.

## FEATURES
snr_list = [0, 5, 10, 15, 20, 25, 30] # list of SNR levels.
input_dim = 257 # number of inputs.
fs = 16000 # sampling frequency (Hz).
Tw = 32 # window length (ms).
Ts = 16 # window shift (ms).
Nw = int(fs*Tw*0.001) # window length (samples).
Ns = int(fs*Ts*0.001) # window shift (samples).
NFFT = int(pow(2, np.ceil(np.log2(Nw)))) # number of DFT components.

## NETWORK AND TRAINING PARAMETERS
mbatch_size = 20 # mini-batch size.
val_interval = 50 # number of mini-batches to complete before validation.
max_mini_batch_count = 100000 # maximum number of mini-batches.
cheat_interval = 2000 # number of mini-batches completed before best model is loaded.
nconst = 32768 # normalization constant.
val_frac = 0.05 # fraction of training waveforms to be used for validation set.
hidden_layers = 5 # number of hidden layers.
cell_units = 1024 # number of units per cell.
num_outputs = input_dim # number of output dimensions.
least_error = float('Inf') # smallest validation error.

## GPU CONFIGURATION
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # select GPU.
config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
config.log_device_placement=False

## PLACEHOLDERS
x_ph = tf.placeholder(tf.int16, shape=[None, None]) # clean placeholder.
d_ph = tf.placeholder(tf.int16, shape=[None, None]) # noise placeholder.
x_len_ph = tf.placeholder(tf.int32, shape=[None]) # clean length placeholder.
d_len_ph = tf.placeholder(tf.int32, shape=[None]) # noise length placeholder.
snr_ph = tf.placeholder(tf.float32, shape=[None]) # SNR placeholder.
train_clean_ph = tf.placeholder(tf.float32, shape=[None, None]) # clean training waveform placeholder.
train_clean_len_ph = tf.placeholder(tf.int32, shape=[None]) # clean training length placeholder.

## DATASETS
print('Preparing datasets...')
train_clean_np, train_clean_len_np = sets.train_set(subset_path + 
	'/train', '*.wav') # clean training waveforms and lengths.
train_noise_np, train_noise_len_np = sets.train_set(noise_path, 
	'*.wav') # noise training waveforms and lengths.
train_snr_np = np.array(snr_list, np.int32) # training snr levels.
val_clean_np, val_clean_len_np, val_snr_np, _ = sets.test_set(subset_path + 
	'/val_clean', '*.wav', snr_list) # clean validation waveforms and lengths.
val_noise_np, val_noise_len_np, _, _ = sets.test_set(subset_path + 
	'/val_noise', '*.wav', snr_list) # noise validation waveforms and lengths.
test_clean_np, test_clean_len_np, test_snr_np, test_fnames_l = sets.test_set(subset_path + 
	'/test_clean', '*.wav', snr_list) # clean test waveforms and lengths.
test_noise_np, test_noise_len_np, _, _ = sets.test_set(subset_path + 
	'/test_noise', '*.wav', snr_list) # noise test waveforms and lengths.

## TRAINING DATASET
train_clean_d = tf.data.Dataset.from_tensor_slices((train_clean_ph, train_clean_len_ph))
train_clean_d = train_clean_d.repeat()
train_clean_d = train_clean_d.shuffle(buffer_size=train_clean_len_np.shape[0])
train_clean_d = train_clean_d.batch(mbatch_size)
train_clean_i = train_clean_d.make_initializable_iterator()
train_clean_g = train_clean_i.get_next()

train_noise_d = tf.data.Dataset.from_tensor_slices((train_noise_np, train_noise_len_np))
train_noise_d = train_noise_d.repeat()
train_noise_d = train_noise_d.shuffle(buffer_size=train_noise_len_np.shape[0])
train_noise_d = train_noise_d.batch(mbatch_size)
train_noise_i = train_noise_d.make_one_shot_iterator()
train_noise_g = train_noise_i.get_next()

train_snr_d = tf.data.Dataset.from_tensor_slices(train_snr_np)
train_snr_d = train_snr_d.repeat()
train_snr_d = train_snr_d.shuffle(buffer_size=train_snr_np.shape[0])
train_snr_d = train_snr_d.batch(mbatch_size)
train_snr_i = train_snr_d.make_one_shot_iterator()
train_snr_g = train_snr_i.get_next()

## FEATURE EXTRACTION FUNCTION
def feat_extr(x, d, x_len, d_len, Q, Nw, Ns, NFFT, fs, P, nconst):
	'''
    Computes magnitude spectrum input features, and the IBM target features. The
	sequences are padded, with seq_len providing the length of each 
	sequence without padding.

    Inputs:
		x - clean waveform (dtype=tf.int32).
		d - noisy waveform (dtype=tf.int32).
		s_len - clean waveform length without padding (samples).
		d_len - noise waveform length without padding (samples).
		Q - SNR level.
		Nw - window length (samples).
		Ns - window shift (samples).
		NFFT - DFT components.
		fs - sampling frequency (Hz).
		P - padded waveform length (samples).
		nconst - normalization constant.

	Outputs:
		x_LSSE - padded noisy LSSEs.
		IBM - padded IBM.	
		seq_len - length of each sequence without padding.
	'''
	(x, y, d) = tf.map_fn(lambda z: feat.addnoisepad(z[0], z[1], z[2], z[3], z[4], 
		P, nconst), (x, d, x_len, d_len, Q), dtype=(tf.float32, tf.float32, 
		tf.float32)) # padded noisy waveform, and padded clean waveform.	
	seq_len = feat.nframes(x_len, Ns) # length of each sequence.
	x_MAG = feat.mag(x, Nw, Ns, NFFT) # clean magnitude spectrum.
	d_MAG = feat.mag(d, Nw, Ns, NFFT) # noise magnitude spectrum.
	y_MAG = feat.mag(y, Nw, Ns, NFFT) # noisy magnitude spectrum.
	IBM = tf.to_float(tf.boolean_mask(tf.greater(x_MAG, d_MAG), tf.sequence_mask(seq_len))) # Ideal Binary Mask (IBM).
	return (y_MAG, IBM, seq_len)


## FEATURE GRAPH
print('Preparing graph...')
P = tf.reduce_max(x_len_ph) # padded waveform length.
feat = feat_extr(x_ph, d_ph, x_len_ph, d_len_ph, snr_ph, Nw, Ns, NFFT, 
	fs, P, nconst) # feature graph.

with tf.name_scope('mask'):
	mask = tf.boolean_mask(feat[0], tf.sequence_mask(feat[2])) # convert from 3D tensor to 2D tensor.

## FNN OPTIONS
fnn_opts = {
	'activation': 'relu'
}

## HIDDEN LAYERS
h = []
for i in range(0,hidden_layers):
	if i is 0:
		h.append(create.fc_layer(mask, input_dim, cell_units, str(i+1), **fnn_opts))
	else:
		h.append(create.fc_layer(h[i-1].output, h[i-1].num_outputs, cell_units, str(i+1), **fnn_opts))

## OUTPUT LAYER
out_opts = {
	'summaries': False,
	'activation': 'affine'
}
y_ = create.fc_layer(h[hidden_layers-1].output, h[hidden_layers-1].num_outputs, 
	num_outputs, 'out', **out_opts)

## ERROR, LOSS, & OPTIMIZER
error = op.error(feat[1], y_.output, 'sigmoid_xentropy')
loss = op.loss(feat[1], y_.output, 'sigmoid_xentropy')
trainer, _ = op.optimizer(loss, optimizer='adam')

## SAVE VARIABLES
saver = tf.train.Saver()

## TRAINING
if training:
	print("Training...")
	with tf.Session(config=config) as sess:
		## VARIABLE INITIALIZER, & TRAINING DATASET INITIALIZER
		sess.run(train_clean_i.initializer, feed_dict={train_clean_ph: train_clean_np,
			train_clean_len_ph: train_clean_len_np}) # initialise clean training waveforms.
		sess.run(tf.global_variables_initializer()) # initialise variables.
		if not os.path.exists('training'):
			os.makedirs('training')
		with open("training/" + version + ".txt", "a") as results:
			results.write("Validation error, mini-batch count, D/T.\n")
		mbatch_count = 0 # number of mini-batches completed.
		while mbatch_count < max_mini_batch_count:
			train_clean_m = sess.run(train_clean_g) # generate mini-batch of clean training waveforms.
			train_noise_m = sess.run(train_noise_g) # generate mini-batch of noise training waveforms.
			train_snr_m = sess.run(train_snr_g) # generate mini-batch of SNR levels.
			sess.run(trainer, feed_dict={x_ph: train_clean_m[0], d_ph: train_noise_m[0], 
				x_len_ph: train_clean_m[1], d_len_ph: train_noise_m[1], snr_ph: train_snr_m}) # training iteration.
			mbatch_count += 1
			if mbatch_count % val_interval == 0:
				val_error = sess.run(error, feed_dict={x_ph: val_clean_np, d_ph: val_noise_np, 
					x_len_ph: val_clean_len_np, d_len_ph: val_noise_len_np, snr_ph: val_snr_np}) # validation error.
				if val_error < least_error:
					least_error = val_error
					saver.save(sess, model_path + "/model.ckpt") # save model if lowest error has been acheived.
				elif mbatch_count % cheat_interval == 0:
					saver.restore(sess, model_path + "/model.ckpt") # restore model that achieved the lowest error.
				print("Training: %3.2f%% complete. Mini-batch count is %d. Validation error: %g." % 
					(100*(mbatch_count/max_mini_batch_count), mbatch_count, val_error), end="\r")
				with open("training/" + version + ".txt", "a") as results:
					results.write("%g, %d, %s.\n" % (val_error, mbatch_count, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

## TEST 
if test:
	print("\nTest...")
	with tf.Session(config=config) as sess:
		saver.restore(sess, model_path + "/model.ckpt")
		for snr in snr_list:
			i = 0 # count.
			correct = 0 # number of correct elements.
			total = 0 # total number of elements.
			snr_path = save_path + '/' + str(snr) + 'dB' # save path for test files at set SNR level.
			if not os.path.exists(snr_path):
				os.makedirs(snr_path)
			for j in range(test_clean_np.shape[0]):
				output = sess.run(y_.output, feed_dict={x_ph: [test_clean_np[j,:]], d_ph: [test_noise_np[j,:]], 
					x_len_ph: [test_clean_len_np[j]], d_len_ph: [test_noise_len_np[j]], snr_ph: [snr]}) # estimate.	
				estimate = np.greater(output, 0) # convert to boolean.
				if save:
					spio.savemat(snr_path + '/' + test_fnames_l[j] + '_' + str(snr) + 'dB', {'IBM_hat':output})
				target = sess.run(feat[1], feed_dict={x_ph: [test_clean_np[j,:]], d_ph: [test_noise_np[j,:]], 
					x_len_ph: [test_clean_len_np[j]], d_len_ph: [test_noise_len_np[j]], snr_ph: [snr]}) # target.	
				target = np.greater(target, 0.5) # convert to boolean.
				correct += np.sum(np.equal(target, estimate)) # add to number of correct elements.
				total += target.size # add to total number of elements.
				i += 1 # increment.
				print('Test: %3.2f%% complete. Correct: %d, Total: %d.' % (100*(i/test_clean_np.shape[0]), 
					correct, total), end="\r")
			with open("accuracy.txt", "a") as acc:
				acc.write("Accuracy: %%%3.2f, version: %s, snr: %d, D/T:%s.\n" % (100*(correct/total), version, snr,
					datetime.now().strftime('%Y-%m-%d %H:%M:%S'))) # accuracy.
			print('\nAccuracy: %%%3.2f, snr: %d.' % (100*(correct/total), snr)) # accuracy.
