# ResBLSTM IBM Estimator

An implementation of a deep residual bidirectional long-short term memory ideal binary mask (ResBLSTM-IBM) estimator in [TensorFlow](https://www.tensorflow.org/). The bidirectional recurrent neural network (BRNN) from [1] has been replaced with a ResBLSTM. The ResBLSTM consists of 5 blocks, with a cell size of 512 for each LSTM cell. The ResBLSTM-IBM estimator can be found [here](https://github.com/anicolson/DeepXi).

## How to Use DeepXi for IBM estimation
The [DeepXi](https://github.com/anicolson/DeepXi) script has a variable called *out_type* on line 54. Set this to 'gain'. Also set the variable *gain* on line 62 to 'ibm', and IBM estimates for given noisy speech will be placed in the *output* directory.

## References
[1] [Nicolson, A. and Paliwal, K.K., 2018. Bidirectional Long-Short Term Memory Network-based Estimation of Reliable Spectral Component Locations. Proc. Interspeech 2018, pp.1606-1610](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1134.pdf)