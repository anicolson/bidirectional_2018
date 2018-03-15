## FILE:           sets.py 
## DATE:           2018
## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
## BRIEF:          Creates training, and test sets from given paths.

import numpy as np
from scipy.io.wavfile import read
import glob, os

def train_set(fdir, fname):
	'''
	Places all of the training waveforms from the list into a numpy array. 
	SPHERE format cannot be used. 'glob' is used to support Unix style pathname 
	pattern expansions. Waveforms are padded to the maximum waveform length. The 
	waveform lengths are recorded so that the correct lengths can be sliced 
	for feature extraction.

	Inputs:
		fdir - directory containing the waveforms.
		fname - filename of the waveforms.

	Outputs:
		wav_np - matrix of paded waveforms stored as a numpy array.
		len_np - length of each waveform strored as a numpy array.
	'''
	wav_l = [] # list for waveforms.
	for fpath in glob.glob(os.path.join(fdir, fname)):
		(_, wav) = read(fpath) # read waveform from given file path.
		wav_l.append(wav) # append.
	len_l = [] # list of the waveform lengths.
	maxlen = max(len(wav) for wav in wav_l) # maximum length of waveforms.
	wav_np = np.zeros([len(wav_l), maxlen], np.int16) # numpy array for waveform matrix.
	for (i, wav) in zip(range(len(wav_l)), wav_l):
		wav_np[i,:len(wav)] = wav # add waveform to numpy array.
		len_l.append(len(wav)) # append length of waveform to list.
	return wav_np, np.array(len_l, np.int32)

def test_set(fdir, fname, snr_l):
	'''
	Places all of the test waveforms from the list into a numpy array. 
	SPHERE format cannot be used. 'glob' is used to support Unix style pathname 
	pattern expansions. Waveforms are padded to the maximum waveform length. The 
	waveform lengths are recorded so that the correct lengths can be sliced 
	for feature extraction. The SNR levels of each test file are placed into a
	numpy array. Also returns a list of the file names.

	Inputs:
		fdir - directory containing the waveforms.
		fname - filename of the waveforms.
		snr_l - list of the SNR levels used.

	Outputs:
		wav_np - matrix of paded waveforms stored as a numpy array.
		len_np - length of each waveform strored as a numpy array.
		snr_test_np - numpy array of all the SNR levels for the test set.
		fnames_l - list of filenames.
	'''
	fnames_l = [] # list of file names.
	wav_l = [] # list for waveforms.
	snr_test_l = [] # list of SNR levels for the test set.
	for fpath in glob.glob(os.path.join(fdir, fname)):
		for snr in snr_l:
			if fpath.find('_' + str(snr) + 'dB') != -1:
				snr_test_l.append(snr) # append SNR level.		
		(_, wav) = read(fpath) # read waveform from given file path.
		wav_l.append(wav) # append.
		fnames_l.append(os.path.basename(os.path.splitext(fpath)[0])) # append name.
	len_l = [] # list of the waveform lengths.
	maxlen = max(len(wav) for wav in wav_l) # maximum length of waveforms.
	wav_np = np.zeros([len(wav_l), maxlen], np.int16) # numpy array for waveform matrix.
	for (i, wav) in zip(range(len(wav_l)), wav_l):
		wav_np[i,:len(wav)] = wav # add waveform to numpy array.
		len_l.append(len(wav)) # append length of waveform to list.
	return wav_np, np.array(len_l, np.int32), np.array(snr_test_l, np.int32), fnames_l

