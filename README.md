# ResLSTM-IBM Estimator
An implementation of a deep Residual Long-Short Term Memory - Ideal Binary Mask (ResBLSTM-IBM) estimator in [TensorFlow](https://www.tensorflow.org/). The Bidirectional Recurrent Neural Network (BRNN) from [1] has been replaced with a ResLSTM. 

## Prerequisites
* [TensorFlow](https://www.tensorflow.org/)
* [Python 3](https://www.python.org/)
* [MATLAB](https://www.mathworks.com/products/matlab.html)

## Download the Model
A trained model can be downloaded from [here](https://www.dropbox.com/s/ecp4a3orzht3j2h/epoch-15.zip?dl=0).

## File Description
File | Description
--------| -----------  
train.py | Training, must give paths to the clean speech and noise training files.
inf.py | Inference, outputs .mat MATLAB IBM estimates.
run.py | Used to pass variables to inf.py. must give paths to the model, and the clean speech and noise testing files.

## References
[1] A. Nicolson and K. K. Paliwal, "Bidirectional Long-Short Term Memory Network-based Estimation of Reliable Spectral Component Locations", Proceedings of Interspeech 2018.
