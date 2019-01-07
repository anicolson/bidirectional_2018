# IBM Estimated Using DeepXi

[DeepXi](https://github.com/anicolson/DeepXi) is now used instead of the bidirectional recurrent neural network (BRNN) from [1]. [DeepXi](https://github.com/anicolson/DeepXi) is a deep residual bidirectional long-short term memory (ResBLSTM) network *a priori* SNR estimator implemented in [TensorFlow](https://www.tensorflow.org/). The *a priori* SNR estimated by [DeepXi](https://github.com/anicolson/DeepXi) is used to compute an ideal binary mask (IBM) estimate. 

## How to Use DeepXi for IBM estimation
The [DeepXi](https://github.com/anicolson/DeepXi) script has a variable called *out_type* on line 54. Set this to 'gain'. Also set the variable *gain* on line 62 to 'ibm', and IBM estimates for given noisy speech will be placed in the *output* directory.

## References
[1] [Nicolson, A. and Paliwal, K.K., 2018. Bidirectional Long-Short Term Memory Network-based Estimation of Reliable Spectral Component Locations. Proc. Interspeech 2018, pp.1606-1610](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1134.pdf)