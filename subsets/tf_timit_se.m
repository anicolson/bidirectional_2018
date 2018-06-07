function tf_timit_se(timit_path, noise_path, save_path, val_frac, Q)
% TF_TIMIT_SE - creates training, validation, and test speech enhancement subsets from the TIMIT corpus for TensorFlow.
%
% Inputs:
%	timit_path - the path for the TIMIT dataset.
%	noise_path - the path to the noise files.
%   save_path - subset save path.
%   val_frac - fraction of test set used as a validation set. 
%	Q - SNR levels.
%
%% FILE:           tf_timit_se.m 
%% DATE:           2018
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Creates training, validation, and test speech enhancement subsets from the TIMIT corpus for TensorFlow.

%% FILE LISTS


%% RECORD INPUTS
d.files = dir(strcat(noise_path,'/*.wav')); % noise files.
train.files = [dir([timit_path,'/timit/train/*/*/si*.wav']); ...
    dir([timit_path,'/timit/train/*/*/sx*.wav'])]; % training files.
test.files = [dir([timit_path,'/timit/test/*/*/si*.wav']); ...
    dir([timit_path,'/timit/test/*/*/sx*.wav'])]; % testing files.

%% CREATE VALIDATION SET
p = randperm(length(train.files), round(length(train.files)*val_frac)); % index of validation files.
val.files = train.files(p); % validation set.
train.files(p) = []; % remove validation files from test set.

%% SAVE DIRECTORIES
if ~exist([save_path, '/train'], 'dir'); mkdir([save_path, '/train']); end % make training set directory.
if ~exist([save_path, '/val_clean'], 'dir'); mkdir([save_path, '/val_clean']); end % make validation set directory.
if ~exist([save_path, '/val_noise'], 'dir'); mkdir([save_path, '/val_noise']); end % make validation set noise directory.
if ~exist([save_path, '/test_clean'], 'dir'); mkdir([save_path, '/test_clean']); end % make test set directory.
if ~exist([save_path, '/test_noise'], 'dir'); mkdir([save_path, '/test_noise']); end % make test set noise directory.

%% VALIDATION SET
p = randperm(length(val.files)); % shuffle validation files.
j = 0; % count.
for i = p
    d.i = randi([1, length(d.files)]); % random noise file.
    d.SNR = Q(randi([1, length(Q)])); % random SNR level.
	[x.wav, fs] = audioread([val.files(i).folder, '/', val.files(i).name]); % read validation clean waveform.
    [d.src, ~] = audioread([d.files(d.i).folder, '/', d.files(d.i).name]); % read validation noise waveform.   
    x.N = length(x.wav); % length of validation clean waveform.
    d.N = length(d.src); % length of validation noise waveform.
    d.R = randi(1 + d.N - x.N);  % generate random start location in noise waveform.
    d.wav = d.src(d.R:d.R + x.N - 1); % noise waveform.    
    [~, spkr, ~] = fileparts(val.files(i).folder); % get speaker.
    audiowrite([save_path, '/val_clean/', spkr, '_', val.files(i).name(1:end-4), ...
        '_', d.files(d.i).name(1:end-4), '_', num2str(d.SNR), 'dB.wav'], x.wav, fs); % write validation clean waveform.
    audiowrite([save_path, '/val_noise/', spkr, '_', val.files(i).name(1:end-4), ...
        '_', d.files(d.i).name(1:end-4), '_', num2str(d.SNR), 'dB.wav'], d.wav, fs); % write validation noise waveform.
    j = j + 1; % increment count.
    clc;
    fprintf('Creating validation set: %3.2f%% complete.\n', 100*(j/length(val.files)));
end

%% TRAINING SET
for i = 1:length(train.files)
	[x.wav, fs] = audioread([train.files(i).folder, '/', train.files(i).name]); % read training waveform.
    [~, spkr, ~] = fileparts(train.files(i).folder); % get speaker.
    audiowrite([save_path, '/train/', spkr, '_', train.files(i).name ], ...
    x.wav, fs); % write training waveform.
    clc;
    fprintf('Creating training set: %3.2f%% complete.\n', 100*(i/length(train.files)));
end

%% TEST SET
p = randperm(length(test.files)); % shuffle test files.
j = 0; % count.
for i = p
    d.i = randi([1, length(d.files)]); % random noise file.
	[x.wav, fs] = audioread([test.files(i).folder, '/', test.files(i).name]); % read test clean waveform.
    [d.src, ~] = audioread([d.files(d.i).folder, '/', d.files(d.i).name]); % read test noise waveform.   
    x.N = length(x.wav); % length of test clean waveform.
    d.N = length(d.src); % length of test noise waveform.
    d.R = randi(1 + d.N - x.N);  % generate random start location in noise waveform.
    d.wav = d.src(d.R:d.R + x.N - 1); % noise waveform.    
    [~, spkr, ~] = fileparts(test.files(i).folder); % get speaker.
    audiowrite([save_path, '/test_clean/', spkr, '_', test.files(i).name(1:end-4), ...
        '_', d.files(d.i).name(1:end-4), '.wav'], x.wav, fs); % write test clean waveform.
    audiowrite([save_path, '/test_noise/', spkr, '_', test.files(i).name(1:end-4), ...
        '_', d.files(d.i).name(1:end-4), '.wav'], d.wav, fs); % write test noise waveform.
    j = j + 1; % increment count.
    clc;
    fprintf('Creating test set: %3.2f%% complete.\n', 100*(j/length(test.files)));
end
end
