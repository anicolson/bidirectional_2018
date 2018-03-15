function timit_ssc_train(timit_path, Tw, Ts, fs, M)
% TIMIT_SSC_TRAIN - train GMM speaker models using clean SSCs.
%
% Inputs:
%	timit_path - the path for the TIMIT dataset.
%	Tw - window length (ms).
%	Ts - window shift (ms).
%	fs - sampling frequency (Hz).
%   M - number of mixtures.
%
%% FILE:           timit_ssc_train.m 
%% DATE:           2017
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Train GMM speaker models using clean SSCs.
	
%% SPEAKER LIST
spkr = dir([timit_path,'/timit/*/*/*']); % list of the speakers.
spkr(ismember({spkr.name}, {'.', '..'})) = []; 

%% RECORD INPUTS
fid = fopen('timit_ssc_par.txt', 'w');
fprintf(fid, 'Tw = %d ms, Ts = %d ms, fs = %d Hz, M = %d\n', Tw, Ts, fs, M); % record inputs.
fprintf(fid, 'TIMIT path: %s\n', timit_path); % record timit path.
fclose(fid);

%% CLEAN
x.Nw = round(fs*Tw*0.001); % window length (samples).
x.Ns = round(fs*Ts*0.001); % window shift (samples).
x.fs = fs; % sampling frequency (Hz).
x.NFFT = 2^nextpow2(x.Nw); % frequency bins (samples).

%% FILTER BANK
[H, bl, bh] = melfbank(26, x.NFFT/2 + 1, fs); % mel-scale filter bank.
       
%% TRAINING GMM
gmm = cell(length(spkr), 1); % storage for each of the speaker models.
for i=1:length(spkr)
    spkr(i).train_files = [dir([timit_path,'/timit/*/*/', spkr(i).name,'/si*.wav']); 
        dir([timit_path,'/timit/*/*/', spkr(i).name,'/sx*.wav'])]; % training files for the speaker.
    x.OBS = cell(length(length(spkr(i).train_files)), 1); % store each of the speaker's observations.
	for j=1:length(spkr(i).train_files)
        [x.wav, ~] = audioread([spkr(i).train_files(j).folder, ...
            '/', spkr(i).train_files(j).name]); % waveform.
        x = ssc_centered(x, H, bl, bh); % compute Spectral Subband Centroids (SSC).
        x.OBS{j} = x.SSC; % store each of the speaker's observations.
    end
    options = statset('Display', 'final', 'UseParallel', true, 'MaxIter', 500);
    gmm{i}.dist = gmdistribution.fit(vertcat(x.OBS{:}), M, 'Start', 'plus', ...
        'CovType', 'diagonal', 'RegularizationValue', 1e-12, 'Options', options); % initialisation using k-means, Expectation Maximisation (EM) training.
	gmm{i}.spkr = spkr(i); % speaker name.
    fprintf('Training features: %d of %d completed.\n', i, length(spkr));
end
save('timit_ssc_gmm.mat', 'gmm');
end
