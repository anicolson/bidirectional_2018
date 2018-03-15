function timit_se(test_path, Tw, Ts, fs, Q)
% TIMIT_SE - speech enhancement with objective scoring (no speech enhancement, i.e. noisy).
%
% Inputs:
%   test_path - the path to the test files.
%   Tw - window length (ms).
%   Ts - window shift (ms).
%   fs - sampling frequency (Hz).
%   Q - SNR values.
%
%% FILE:           timit_se.m 
%% DATE:           2018
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Speech enhancement with objective scoring (no speech enhancement, i.e. noisy).
	
%% FILE LISTS
x.files = dir([test_path, '/test_clean/*.wav']); % test clean files.
d.files = dir([test_path, '/test_noise/*.wav']); % test noise files.

%% RECORD INPUTS
fid = fopen('par.txt', 'w');
fprintf(fid, 'Tw = %d ms, Ts = %d ms, fs = %d Hz\n', ...
    Tw, Ts, fs); % record inputs.
fprintf(fid, 'Test path: %s\nSNR values (dB): ', test_path); % record paths.
for i = 1:length(Q); fprintf(fid, '%g ', Q(i)); end % dB values.
fclose(fid);

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
fid1 = fopen(strcat('indi.txt'), 'w'); % individual test results.
fid2 = fopen(strcat('avg.txt'), 'w'); % average test results.
for i=1:length(Q)
    avgSNR = 0; % average SNR for ideal case.
    avgSegSNR = 0; % average segmental SNR for ideal case.
    avgWPESQ = 0; % average WPESQ for ideal case.
    avgQSTI = 0; % average QSTI for ideal case.
    for j=1:length(x.files)
        x.wav = x.files(j).wav; % clean waveform.
        d.wav = d.files(j).wav; % noise waveform.
        [y.wav, ~] = addnoise(x.wav, d.wav, Q(i)); % noisy waveform.
        
        %% NOISY RESULTS
        SNR = segsnr(x.wav, y.wav, fs); % find SegSNR and SNR.
        WPESQ = pesqbin(x.wav, y.wav, fs, 'wb'); % find Wideband PESQ
        QSTI = qsti(x.wav, y.wav, fs); % find QSTI.
                                
        %% RECORD RESULTS
        fprintf(fid1, 'file: %s.\nSNR: %3.2f dB, segSNR: %3.2f, WPESQ: %1.2f, QSTI: %1.2f.\n', x.files(j).name, ...
            SNR.SNR, SNR.SNRseg, WPESQ, QSTI); % individual test results.
        avgSNR = avgSNR + SNR.SNR; % sum of all SNR levels.
        avgSegSNR = avgSegSNR + SNR.SNRseg; % sum of all segmental SNR levels. 
        avgWPESQ = avgWPESQ + WPESQ; % sum of all WPESQ values.
        avgQSTI = avgQSTI + QSTI; % sum of all QSTI values.

        clc;
        fprintf('Percentage complete for %ddB: %3.2f%%.\n', Q(i), 100*(j/length(x.files)));
    end
    avgSNR = avgSNR/length(x.files);
    avgSegSNR = avgSegSNR/length(x.files);
    avgWPESQ = avgWPESQ/length(x.files);
    avgQSTI = avgQSTI/length(x.files);
    fprintf(fid2, 'Av. SNR: %2.2fdB, av. SegSNR: %2.2fdB, av. WPESQ: %1.2f, av. QSTI: %1.2f, SNR: %ddB.\n', ...
        avgSNR, avgSegSNR, avgWPESQ, avgQSTI, Q(i)); % average results.
end
fclose(fid1);
fclose(fid2);
end
