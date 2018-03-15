function x = analysis_mag(x)
% ANALYSIS_MAG - magnitude spectrum of a signal for analysis.
%
% Inputs:
%	x.wav - input sequence.
%	x.Nw - frame width (samples).
%	x.Ns - frame shift (samples).
%	x.NFFT - number of frequency bins.
%
% Outputs:
%	x.frm - framing & windowing.
%	x.MAG - magnitude spectrum.
%	x.PHA - phase spectrum.

%% FILE:           analysis_mag.m 
%% DATE:           2017
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Computes the magnitude spectrum of a waveform for analysis.

x.frm = frame(x.wav, x.Nw, x.Ns); % framing & windowing.
x.DFT = fft(x.frm, x.NFFT, 2); % complex short-time DFT.
x.MAG = abs(x.DFT); % magnitude spectrum.
x.MAG = x.MAG(:,1:x.NFFT/2 + 1); % single-sided magnitude spectrum.
x.PHA = angle(x.DFT); % phase spectrum.
end
%% EOF
