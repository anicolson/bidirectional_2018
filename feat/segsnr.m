function SNR = segsnr(s, y, Fs, seglen, thresh)
% SEGSNR - computes segmental SNR of a signal.
%
% NOTE: it is assumed that both signals are in phase with each other.
%
% Inputs:
%	s - original, clean signal.
%	y - the enhanced signal.
%	Fs - sampling frequency.
%	seglen - time length of each segment (ms).
%	thresh - SNRseg thresh dB, to avoid large negative SNRseg values.
%
% Outputs:
%	SNR - structure containing SNRseg, SNRseg_R and SNR.

%% FILE:           segsnr.m 
%% DATE:           2017
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Calculates SNR, SNRseg and SNRseg_R.
%% VERSION:        MATLAB R2013a

if     nargin == 2, thresh = -10; seglen = 20; Fs = 8000; 
elseif nargin == 3, thresh = -10; seglen = 20; 
elseif nargin == 4, thresh = -10;
end

y = y(:)';
s = s(:)';

% original and processed signal must be the same length.
if length(y) ~= length(s); 
	error('snr.m: lengths do not match.\n');
end

N = (Fs*seglen)/1000;                 % segment length.
M = floor(length(s)/N);               % number of segments.
n = (repmat((1:N:M*N)',1,N));         % segment indexes.
n = n + repmat(0:(N-1),M,1);

%% NOISE
d = s - y; % difference between the signals.

%% SNRseg
SNR.SNRseg = 10*log10((sum(s(n).^2,2)./(sum(d(n).^2,2) + eps)));
SNR.SNRseg = SNR.SNRseg(SNR.SNRseg > thresh);
SNR.SNRseg = mean(SNR.SNRseg);

%% SNRseg_R 
% alleviates large negative SNRseg values caused by the signal 
% energy during silenced segments.
SNR.SNRseg_R = mean(10*log10(1+(sum(s(n).^2,2)./(sum(d(n).^2,2) + eps))));

%% SNR
SNR.SNR = 10*log10(sum(s.^2)/sum((d).^2)); 
end
%%EOF
