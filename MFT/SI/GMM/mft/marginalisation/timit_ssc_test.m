function timit_ssc_test(test_path, model_path, Tw, Ts, fs, Q)
% TIMIT_SSC_TEST - tests GMM speaker models using marginalization and SSCs.
%
% Inputs:
%	test_path - the path to the test files.
%	model_path - path to the GMM speaker models.
%	Tw - window length (ms).
%	Ts - window shift (ms).
%	fs - sampling frequency (Hz).
%	Q - SNR values.
%
%% FILE:           timit_ssc_test.m 
%% DATE:           2018
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Tests GMM speaker models using marginalization and SSCs.

%% FILE LISTS
x.files = dir([test_path, '/test_clean/*.wav']); % test clean files.
d.files = dir([test_path, '/test_noise/*.wav']); % test noise files.

%% RECORD INPUTS
fid = fopen('timit_ssc_par.txt', 'w');
fprintf(fid, 'Tw = %d ms, Ts = %d ms, fs = %d Hz\n', ...
    Tw, Ts, fs); % record inputs.
fprintf(fid, 'Test path: %s\nModel path: %s\nSNR values (dB): ', ...
    test_path, model_path); % record paths.
for i = 1:length(Q); fprintf(fid, '%g ', Q(i)); end % dB values.
fclose(fid);

%% CLEAN
x.Nw = round(fs*Tw*0.001); % window length (samples).
x.Ns = round(fs*Ts*0.001); % window shift (samples).
x.fs = fs; % sampling frequency (Hz).
x.NFFT = 2^nextpow2(x.Nw); % frequency bins (samples).

%% NOISE
d.Nw = round(fs*Tw*0.001); % window length (samples).
d.Ns = round(fs*Ts*0.001); % window shift (samples).
d.fs = fs; % sampling frequency (Hz).
d.NFFT = 2^nextpow2(d.Nw); % frequency bins (samples).

%% NOISY
y.Nw = round(fs*Tw*0.001); % window length (samples).
y.Ns = round(fs*Ts*0.001); % window shift (samples).
y.fs = fs; % sampling frequency (Hz).
y.NFFT = 2^nextpow2(y.Nw); % frequency bins (samples).

%% FILTER BANK
[H, bl, bh] = melfbank(26, y.NFFT/2 + 1, fs); % mel-scale filter bank.

%% LOAD TEST SPEECH INTO MEMORY
for i=1:length(x.files)
    x.files(i).wav = audioread([x.files(i).folder, ...
            '/', x.files(i).name]); % clean test waveform.
    C = strsplit(x.files(i).name, '_'); % use strsplit to get speaker name.
    x.files(i).name = C{1}; % speaker name.
    x.files(i).file = C{2}; % file name.
end

%% LOAD TEST NOISE INTO MEMORY
for i=1:length(d.files)
    d.files(i).wav = audioread([d.files(i).folder, ...
            '/', d.files(i).name]); % noise test waveform.
    C = strsplit(d.files(i).name, '_'); % use strsplit to get noise name.
    d.files(i).name = C{3}; % noise name.
end

%% NOISY TEST GMM
load(model_path);
fid1 = fopen(strcat('timit_ssc_indi.txt'), 'w'); % individual test results.
fid2 = fopen(strcat('timit_ssc_acc.txt'), 'w'); % accuracy test results.
for i=1:length(Q)
	test.total = 0; % total testing files.
	test.correct = 0; % correctly classified.
	for j=1:length(x.files)
        x.wav = x.files(j).wav; % clean waveform.
        d.src = d.files(j).wav; % noise waveform.
        [y.wav, d.wav] = addnoise(x.wav, d.src, Q(i)); % noisy waveform.
        y = ssc_centered(y, H, bl, bh); % compute noisy Spectral Subband Centroids (SSC).
        x = lsse(x, H); % clean Log Spectral Subband Energies (LSSE).
        d = lsse(d, H); % noise Log Spectral Subband Energies (LSSE).
        IBM = x.LSSE > d.LSSE; % IBM with 0 dB threshold.
        seq_log_like = zeros(1, length(gmm)); % sequence log-likelihood. 
        for m=1:length(gmm)
            seq_log_like(m) = marg_mft_diag_gmm_seq_log_like(y.SSC, IBM, gmm{m}.dist.mu, ...
                gmm{m}.dist.Sigma, gmm{m}.dist.ComponentProportion); % compute marginal sequence log-likelihood.
        end
        clc;
        fprintf('True spkr: %s, file: %s, noise: %s, dB: %g, %3.2f%% complete.\n', ...
            x.files(j).name, x.files(j).file, d.files(j).name, Q(i), 100*(j/length(x.files)));
        [~, pred] = max(seq_log_like); % predicted speaker.
        fprintf(fid1, 'pred: %s, true: %s, file: %s, noise: %s, level: %g dB.\n', gmm{pred}.spkr.name, ...
             x.files(j).name, x.files(j).file, d.files(j).name, Q(i)); % individual test results.
        test.total = test.total + 1; % add to test total.
        if gmm{pred}.spkr.name == x.files(j).name
            test.correct = test.correct + 1; % correctly classified.
        end
	end
	fprintf('Speaker identification accuracy at %g dB: %2.2f%%\n', ...
        Q(i), 100*(test.correct/test.total));
	fprintf(fid2, 'Speaker identification accuracy at %g dB: %2.2f%%\n', ...
        Q(i), 100*(test.correct/test.total));
end
fclose(fid1);
fclose(fid2);
end
