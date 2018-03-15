function timit_ssc_test(timit_path, model_path, Tw, Ts, fs)
% TIMIT_SSC_TEST - tests GMM speaker models using clean SSCs.
%
% Inputs:
%	timit_path - the path for the TIMIT dataset.
%	model_path - path to the GMM speaker models.
%	Tw - window length (ms).
%	Ts - window shift (ms).
%	fs - sampling frequency (Hz).
%
%% FILE:           timit_ssc_test.m 
%% DATE:           2017
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Tests GMM speaker models using clean SSCs.
	
%% SPEAKER LIST
spkr = dir([timit_path,'/timit/*/*/*']); % list of the speakers.
spkr(ismember({spkr.name}, {'.', '..'})) = []; 

%% RECORD INPUTS
fid = fopen(strcat('timit_ssc_par.txt'), 'w');
fprintf(fid, 'Tw = %d ms, Ts = %d ms, fs = %d Hz\n', ...
    Tw, Ts, fs); % record inputs.
fprintf(fid, 'TIMIT path: %s\nmodel path: %s\n', timit_path, model_path); % record timit/model path.
fclose(fid);

%% CLEAN
x.Nw = round(fs*Tw*0.001); % window length (samples).
x.Ns = round(fs*Ts*0.001); % window shift (samples).
x.fs = fs; % sampling frequency (Hz).
x.NFFT = 2^nextpow2(x.Nw); % frequency bins (samples).

%% FILTER BANK
[H, bl, bh] = melfbank(26, x.NFFT/2 + 1, fs); % mel-scale filter banks.
       
%% CLEAN TEST GMM
test.total = 0; % total testing files.
test.correct = 0; % correctly classified.
fid = fopen(strcat('timit_ssc_indi.txt'), 'w'); % individual test results.
load(model_path);
for i=1:length(spkr)
	spkr(i).test_files = dir([timit_path,'/timit/*/*/', spkr(i).name,'/sa*.wav']); % testing files for the speaker.
    for j=1:length(spkr(i).test_files)
        [x.wav, ~] = audioread([spkr(i).test_files(j).folder, '/', spkr(i).test_files(j).name]); % waveform.
        x = ssc_centered(x, H, bl, bh); % compute Spectral Subband Centroids (SSC).
        seq_log_like = zeros(1, length(gmm)); % sequence log-likelihood. 
        for k=1:length(gmm)
            seq_log_like(k) = diag_gmm_seq_log_like(x.SSC, gmm{k}.dist.mu, ...
                gmm{k}.dist.Sigma, gmm{k}.dist.ComponentProportion);  
        end
        clc;
        fprintf('Computing sequence log-likelihood. True speaker: %i, file: %i.\n', i, j);
        [~, pred] = max(seq_log_like); % predicted speaker.
        fprintf(fid, 'pred: %s, true: %s, file: %s.\n', gmm{pred}.spkr.name, ...
            spkr(i).name, spkr(i).test_files(j).name); % individual test results.
        test.total = test.total + 1; % add to test total.
        if gmm{pred}.spkr.name == spkr(i).name
            test.correct = test.correct + 1; % correctly classified.
        end
    end
end
fclose(fid);
fid = fopen(strcat('timit_ssc_acc.txt'), 'w'); % accuracy test results.
fprintf('Speaker identification accuracy: %2.2f%%\n', 100*(test.correct/test.total));
fprintf(fid, 'Speaker identification accuracy: %2.2f%%\n', 100*(test.correct/test.total));
fclose(fid);
end
