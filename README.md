# ResBLSTM-IBM Estimator
An implementation of a deep Residual Bidirectional Long-Short Term Memory - Ideal Binary Mask (ResBLSTM-IBM) estimator in [TensorFlow](https://www.tensorflow.org/). The Bidirectional Recurrent Neural Network (BRNN) from [1] has been replaced with a ResBLSTM. 

## Prerequisites
* [TensorFlow](https://www.tensorflow.org/)
* [Python 3](https://www.python.org/)
* [MATLAB](https://www.mathworks.com/products/matlab.html)

File | Description
--------| -----------  
MFT | The IBM Estimators and the [marginalisation-based ASI](https://maxwell.ict.griffith.edu.au/spl/publications/papers/icsps17_aaron.pdf) system.
train.py | Training, must give paths to the clean speech and noise training files.
inf.py | Inference, outputs .mat MATLAB IBM estimates.
run.py | Used to pass variables to inf.py. must give paths to the model, and the clean speech and noise testing files.

## Training and testing the BRNN-IBM Estimator for it IBM estimate accuracy:
* Create the training, validation, and testing subsets uing functions in the [subsets](https://github.com/anicolson/bidirectional_2018/tree/master/subsets) directory.
* Run the corresponding brnn.py python3 script for either the [speech enhancement dataset](https://github.com/anicolson/bidirectional_2018/tree/master/MFT/IBM/IBM_hat/BRNN/SE/TIMIT/MAG/brnn.py) or the [speaker identification dataset](https://github.com/anicolson/bidirectional_2018/blob/master/MFT/IBM/IBM_hat/BRNN/SI/TIMIT/LSSE/brnn.py). The IBM estimation accuracy will be stored in [accuracy.txt](https://github.com/anicolson/bidirectional_2018/blob/master/MFT/IBM/IBM_hat/BRNN/SI/TIMIT/LSSE/accuracy.txt).

## References
[1] A. Nicolson and K. K. Paliwal, "Bidirectional Long-Short Term Memory Network-based Estimation of Reliable Spectral Component Locations", Proceedings of Interspeech 2018.
