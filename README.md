# IBM Estimated Using Deep Xi

[Deep Xi](https://github.com/anicolson/DeepXi) from [1] is now used instead of the bidirectional recurrent neural network (BRNN) from [2]. [Deep Xi](https://github.com/anicolson/DeepXi) is a deep learning approach to *a priori* SNR estimation, implemented in [TensorFlow](https://www.tensorflow.org/). The *a priori* SNR estimated by [Deep Xi](https://github.com/anicolson/DeepXi) is used to compute an ideal binary mask (IBM) estimate. 

**[Deep Xi](https://github.com/anicolson/DeepXi) can be found [here](https://github.com/anicolson/DeepXi).**

## References

https://doi.org/10.1016/j.specom.2019.06.002


[1] [A. Nicolson and K. K. Paliwal, "Deep Learning For Minimum Mean-Square Error Approaches to Speech Enhancement", Speech Communication, 2019, ISSN 0167-6393, https://doi.org/10.1016/j.specom.2019.06.002.](https://doi.org/10.1016/j.specom.2019.06.002)
[2] [Nicolson, A. and Paliwal, K.K., 2018. Bidirectional Long-Short Term Memory Network-based Estimation of Reliable Spectral Component Locations. Proc. Interspeech 2018, pp.1606-1610](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1134.pdf)
