function [x] = synthesis_mag(x)
% SYNTHESIS_MAG - computes waveform from magnitude spectrum.
%
% Inputs:
%	x.MAG - magnitude spectrum.
%	x.PHA - phase spectrum.
%	x.N - signal length (samples).
%	x.Nw - frame width (samples).
%	x.Ns - frame shift (samples).
%	x.NFFT - number of frequency bins.
%
% Outputs:
%	x.MAG - magnitude spectrum.
%	x.DFT - DFT.
%	x.wav - reconstructed waveform.

%% FILE:           synthesis_mag.m 
%% DATE:           2017
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Computes waveform from magnitude spectrum (synthesis).

x.MAG = [x.MAG, fliplr(x.MAG(:,2:end-1))]; % undo single-sided magnitude spectrum.
x.DFT = x.MAG.*exp(1i*x.PHA); % recombine the magnitude and phase to produce the modified STFT.
x.wav = real(ifft(x.DFT, x.NFFT, 2)); % perform inverse STFT analysis (discard imaginary time domain components).
x.wav = x.wav(:, 1:x.Nw); % discard FFT padding from frames.
x.wav = overlap_add(x.wav, x.N, x.Nw, x.Ns); % frame the input signal, and window each frame.
end
%% EOF
