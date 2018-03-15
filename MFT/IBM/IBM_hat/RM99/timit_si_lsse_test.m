function timit_si_lsse_test(test_path, Tw, Ts, fs, Q)
% TIMIT_SI_LSSE_TEST - tests LSSE-IBM estimation from [1] for the SI set.
%
% Inputs:
%	test_path - the path to the test files.
%	Tw - window length (ms).
%	Ts - window shift (ms).
%	fs - sampling frequency (Hz).
%	Q - SNR values.
%
%% FILE:           timit_si_lsse_test.m
%% DATE:           2018
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Tests LSSE IBM estimation from [1] for the SI set.

%% FILE LISTS
x.files = dir([test_path, '/test_clean/*.wav']); % test clean files.
d.files = dir([test_path, '/test_noise/*.wav']); % test noise files.

%% RECORD INPUTS
fid = fopen('timit_si_lsse_par.txt', 'w');
fprintf(fid, 'Tw = %d ms, Ts = %d ms, fs = %d Hz\n', ...
    Tw, Ts, fs); % record inputs.
fprintf(fid, 'Test path: %s\nSNR values (dB): ', test_path); % record paths.
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
[H, ~, ~] = melfbank(26, y.NFFT/2 + 1, fs); % mel filter bank.

%% LOAD TEST SPEECH INTO MEMORY
for i=1:length(x.files)
    x.files(i).wav = audioread([x.files(i).folder, ...
            '/', x.files(i).name]); % clean test waveform.
end

%% LOAD TEST NOISE INTO MEMORY
for i=1:length(d.files)
    d.files(i).wav = audioread([d.files(i).folder, ...
            '/', d.files(i).name]); % noise test waveform.
end

%% NOISY TEST
fid1 = fopen(strcat('timit_si_lsse_acc.txt'), 'w'); % individual test results.
for i=1:length(Q)
    total = 0; % total components.
    correct = 0; % correct components.
    for j=1:length(x.files)
        x.wav = x.files(j).wav; % clean waveform.
        d.wav = d.files(j).wav; % noise waveform.
        [y.wav, ~] = addnoise(x.wav, d.wav, Q(i)); % noisy waveform.
        x = lsse(x, H); % clean LSSEs.
        d = lsse(d, H); % noise LSSEs.
        y = lsse(y, H); % noisy LSSEs.

        %% IBM
        ideal.IBM = x.LSSE > d.LSSE; % IBM with 0 dB threshold.   

        %% IBM ESTIMATE - RM99
        est.IBM = rm99(y.SSE, 5); % first 5 frames used for noise estimation.

        %% ACCURACY
        correct = correct + sum(sum(ideal.IBM == est.IBM));
        total = total + size(ideal.IBM, 1)*size(ideal.IBM, 2);

        clc;
        fprintf('Percentage complete for %ddB: %3.2f%%.\n', Q(i), 100*(j/length(x.files)));
    end
    fprintf(fid1, 'Accuracy %3.2f%%, SNR: %ddB.\n', ...
        (correct/total)*100, Q(i)); % average results.
end
fclose(fid1);
end