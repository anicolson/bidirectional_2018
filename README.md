# BRNN-IBM Estimator

The implementation of the Bidirectional Recurrent Neural Network - Ideal Binary Mask (BRNN-IBM) estimator in [TensorFlow](https://www.tensorflow.org/). The BRNN uses Long-Short Term Memory (LSTM) cells. 

## Prerequisites
* [TensorFlow](https://www.tensorflow.org/)
* [Python 3](https://www.python.org/)
* [MATLAB R2017a](https://au.mathworks.com/products/matlab.htmll)
* [VOICEBOX](http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html)
* [addnoise](https://au.mathworks.com/matlabcentral/fileexchange/32136-add-noise?focused=5193299&tab=function)
* [WPESQ](https://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en)
* [PESQ MATLAB Wrapper](https://au.mathworks.com/matlabcentral/fileexchange/33820-pesq-matlab-wrapper)



Directory | Description
--------| -----------  
MFT | The IBM Estimators and the [marginalisation-based ASI](https://maxwell.ict.griffith.edu.au/spl/publications/papers/icsps17_aaron.pdf) system.
SE | Speech Enhancement methods.
tf | Functions for creating [TensorFlow](https://www.tensorflow.org/) graphs and reading the data subsets.
feat | MATLAB feature creation functions from [matlab_feat](https://github.com/anicolson/matlab_feat).
subsets | Creates data subsets from the [TIMIT](https://catalog.ldc.upenn.edu/ldc93s1) speech dataset, and the [RSG-10](https://catalog.ldc.upenn.edu/ldc93s1http://www.steeneken.nl/wp-content/uploads/2014/04/RSG-10_Noise-data-base.pdf) noise dataset.

