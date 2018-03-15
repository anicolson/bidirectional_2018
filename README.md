# BRNN-IBM Estimator

The implementation of the Bidirectional Recurrent Neural Network - Ideal Binary Mask (BRNN-IBM) estimator in [TensorFlow](https://www.tensorflow.org/). The BRNN uses Long-Short Term Memory (LSTM) cells. 

## Prerequisites
* [TensorFlow](https://www.tensorflow.org/)
* [Python 3](https://www.python.org/)
* [MATLAB R2017a](https://au.mathworks.com/products/matlab.htmll)
* [addnoise](https://au.mathworks.com/matlabcentral/fileexchange/32136-add-noise?focused=5193299&tab=function)
* [VOICEBOX](http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html) (for speech enhancement)
* [WPESQ](https://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en) (for speech enhancement)
* [PESQ MATLAB Wrapper](https://au.mathworks.com/matlabcentral/fileexchange/33820-pesq-matlab-wrapper) (for speech enhancement)

Directory | Description
--------| -----------  
MFT | The IBM Estimators and the [marginalisation-based ASI](https://maxwell.ict.griffith.edu.au/spl/publications/papers/icsps17_aaron.pdf) system.
SE | Speech Enhancement methods.
tf | Functions for creating [TensorFlow](https://www.tensorflow.org/) graphs and reading the data subsets.
feat | MATLAB feature creation functions from [matlab_feat](https://github.com/anicolson/matlab_feat).
subsets | Creates data subsets from the [TIMIT](https://catalog.ldc.upenn.edu/ldc93s1) speech dataset, and the [RSG-10](https://catalog.ldc.upenn.edu/ldc93s1http://www.steeneken.nl/wp-content/uploads/2014/04/RSG-10_Noise-data-base.pdf) noise dataset.

## Training and testing the BRNN-IBM Estimator for it IBM estimate accuracy:
* Create the training, validation, and testing subsets uing functions in the [subsets](https://github.com/anicolson/bidirectional_2018/tree/master/subsets) directory.
* Run the corresponding brnn.py python3 script for either the [speech enhancement dataset](https://github.com/anicolson/bidirectional_2018/tree/master/MFT/IBM/IBM_hat/BRNN/SE/TIMIT/MAG/brnn.py) or the [speaker identification dataset](https://github.com/anicolson/bidirectional_2018/blob/master/MFT/IBM/IBM_hat/BRNN/SI/TIMIT/LSSE/brnn.py). The IBM estimation accuracy will be stored in [accuracy.txt](https://github.com/anicolson/bidirectional_2018/blob/master/MFT/IBM/IBM_hat/BRNN/SI/TIMIT/LSSE/accuracy.txt).
