import inf

gpu = 0
epoch = 5
snr = [-10, -5, 0, 5, 10, 15, 20]
test_clean = ''
test_noise = '' 
out_path = '' + str(epoch) 
model_path = ''

inf.IBM_HAT(test_clean, test_noise, out_path, model_path, epoch, snr, gpu)
