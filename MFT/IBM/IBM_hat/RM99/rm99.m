function IBM_hat = rm99(Y, n)
% RM99 - ideal binary mask estimator computed using Gaussian noise statistics based on [1].
%
% Inputs:
%	Y - noisy single-sided magnitude spectrum.
%	n - number of frames to use for noise estimation.
%
% Output:
%	IBM_hat - IBM estimate.
%
%% FILE:           rm99.m 
%% DATE:           2017
%% AUTHOR:         Aaron Nicolson
%% AFFILIATION:    Signal Processing Laboratory, Griffith University
%% BRIEF:          Ideal Binary Mask (IBM) estimator computed using Gaussian noise statistics based on [1].
%
%% REFERENCE:
%	[1] Renevey, P. and Drygajlo, A., 1999. Missing feature theory and 
%	probabilistic estimation of clean speech components for robust speech 
%	recognition. In Sixth European Conference on Speech Communication and 
%	Technology.

mu_n = mean(Y(1:n,:), 1); % mean of the noise magnitude spectrum.
mu_x = Y - mu_n; % mean of the estimated original speech.
IBM_hat = mu_x > Y - mu_x; % 0 dB threshold.
end
