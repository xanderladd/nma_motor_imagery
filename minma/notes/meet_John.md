# Discussions with John


- 4 class classification should have confusion matrix

## W2D3 20th July
* Feature extraction:
    - pick electrodes using GLM: trial-by-trial? or full-time course?
    - pick time windows: 150 ms
    - PSD feature in decoding:
        - chop 3 sec trials into several 150 ms windows, to increase sample size?
        - 150 ms sliding time window for PSD calculation across all electrodes
        - power vs. frequency
        - PSD over HFB (76hz-100hz) a vector of electrode
        - feature for decoder
    - Filter using moving average over amplitudes
        - average 150 time points
        - kinda like smoothing?
    - Butterworth filtering over amplitudes: help us pick the amps in a particular freq band

* John's comments
    - pick electrodes during decoding:
        - add regularization term that would basically pick electrodes automatically?
    - logistic regression with L1 regularisation or combo of L1 and L2
        - std discrete classifier
    - for PSD:
        - not for each of 76 - 100 hz freq - too high feature to sample ratio
        - compare power spectra for two event types and pick the freq band that shows most difference
        - something about integrating over a full band?
        - look into LFBs as well
        - 150 ms time windows won't all be the same in terms of neural signal for a given event type
    - moving average:
        - over-represent low frequency signals
        - surprising if it works!
    - Butterworth:
        - kinda similar to PSD
        - so either PSD or Butterworth
        - after butter you get a high freq signal
            - extract the envelope of the change
            - rectifying with low pass filter

    - compute a spectrogram
        - matrix of values across time (3000/150) x freq (150 frequencies [0Hz - 500Hz, uniformly spaced])
        - might need a larger window for lower freqs
        - compare spectrograms for different event types

    - baselining:
        - good way to improve signal-to-noise ratio
        - z-scoring
        - subtracting the mean
        - dividing by the mean

    - decoding rest?

    - avoid using too many features - couple hundred should be good

    - use HFB for each electrode
    - then decoding, finding the accuracy
    - updating to improve accuracy

    - John's paper: https://www.sciencedirect.com/science/article/pii/S105381191830332X

    - work on the abstract, send the write-up
    - meet tomorrow maybe around 1:45 am
