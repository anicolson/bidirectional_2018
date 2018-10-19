import inf

gpu = 0 # GPU number to use.
epoch = 15 # use model from this epoch of training.
test_noisy = '/home/aaron/set/recon_2018/test/noisy_speech' # path to noisy speech .wav files.
out_path = '/home/aaron' # path to the output directory
model_path = '/media/aaron/Filesystem/model/DeepXi' # path to the directory of the model.
opt = 'ibm' # 'y' for enhanced speech .wav file output, 'ibm' for estimated IBM .mat file output.

inf.DeepXi(test_noisy, out_path, model_path, epoch, gpu, opt)
