# ResLSTM-IBM Estimator
An implementation of a deep Residual Long-Short Term Memory - Ideal Binary Mask (ResBLSTM-IBM) estimator in [TensorFlow](https://www.tensorflow.org/). The Bidirectional Recurrent Neural Network (BRNN) from [1] has been replaced with a ResLSTM. 

## Prerequisites
* [TensorFlow](https://www.tensorflow.org/)
* [Python 3](https://www.python.org/)
* [MATLAB](https://www.mathworks.com/products/matlab.html)

## Download the Model
A trained model can be downloaded from [here](https://www.dropbox.com/s/ecp4a3orzht3j2h/epoch-15.zip?dl=0). The model was trained with a sampling rate of 16 kHz.

## Training 
The following clean speech and noise was used to train the given model:

### Clean Speech:
- The *train-clean-100* set from the the Librispeech corpus (28,539 utterances).
- The CSTR VCTK Corpus (42,015 utterances).
- The *si* *sx* training sets from the TIMIT corpus (3,696 utterances).

### Noise:
- The QUT-NOISE dataset. 
- The Nonspeech dataset.
- The Environmental Background Noise dataset.
- The noise set from the MUSAN corpus.
- Multiple [FreeSound](https://freesound.org/) packs (147, 199, 247, 379, 622, 643, 1,133, 1,563, 1,840, 2,432, 4,366, 4,439, 15,046, 15,598, 21,558). 
- Coloured noise (with an alpha value ranging from -2 to 2 in increments of 0.25).

## File Description
File | Description
--------| -----------  
train.py | Training, must give paths to the clean speech and noise training files.
inf.py | Inference, outputs .mat MATLAB IBM estimates.
run.py | Used to pass variables to inf.py. must give paths to the model, and the clean speech and noise testing files.

## References
[1] A. Nicolson and K. K. Paliwal, "Bidirectional Long-Short Term Memory Network-based Estimation of Reliable Spectral Component Locations", Proceedings of Interspeech 2018.
