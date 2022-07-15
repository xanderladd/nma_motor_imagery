From Cortical activity during motor execution, motor imagery, and imagery-based online feedback by Kai Miller and Rajesh P.N. Rao

- A high-frequency band (HFB) (76–100 Hz) and a low- frequency band (LFB) (8–32 Hz) were chosen for analysis.

> Miller et al.
>> 38 electrodes were selected to quantify the mag- nitude of spectral change: those in which there was a significant difference in the PSD for both the HFB and LFB associated with movement (P < 0.05 t test, Bonferroni-corrected for the number of electrodes in that subject; 21 electrodes for tongue, 17 for hand). In these electrodes, the magnitude of spectral change in the HFB for imagery was 26% of that during actual movement (Fig. 2). For the LFB, the relative magnitude was 49%. The LFB change was significantly larger than the HFB change (P = 0.005 by permutation resampling, 105 iterations).

- Recpie for calculating all the info about P-Value electrodes used in the paper of the above portion:
1. Assume starting point are:
    - Design matrix of electrode recording $X$, where $x_{ti}$ is a scalar value representing electrode $i$ at time $t$. Then also $\bf{x_i}$ is the entire timeseries for electrode i.
    - Calculate PSD for LFB and HFB using Welch's method from NeuroDSP. Welch's method defined as:
    \begin{equation}
        x_m(n) \triangleq = w(n)x(n + mR) where $n = 1, ..., M-1 $ and $m = 0, ..., K-1$
    \end{equaation}
    where **$x$ is actually x_i$ or the time series for one electrode. m denotes a specific window, R is window size and K is number of possible windows. Periodgram for the m-th block given by:
    \begin{equation}
        P_{x_m,M}(\omega_k) = \frac{1}{M}|FFT_{N,k}(x_m)|^2 \deltaeq  \frac{1}{M}|\sum^{N-1}_{n=0} x_m(n)e^{j2\pi nk / M}|^2
    \end{equation}
    - then this is finished by averaging over windows:
    \begin{equation}
        \hat{S}_{x}^W (\omega_k) = frac{1}{K}\sum_{m=0}^{K-1} P_{x_m,M} (\omega_k) 
    \end{equation}
    [SOURCE](https://ccrma.stanford.edu/~jos/sasp/Welch_s_Method.html#:~:text=Welch's%20method%20%5B297%5D%20(also,for%20each%20block%2C%20and%20averaging.&text=is%20the%20rectangular%20window%2C%20the,overlapping%20successive%20blocks%20of%20data.)

## TODO: look into what $\omega_k$ 