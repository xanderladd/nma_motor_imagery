## Questions
Q. How precisely/accurately our model can classify motor behavior among - real, imagery, hand and tongue movements from ECoG neural activity?
(2. Can we use the model to show ***how*** electrodes/regions are more/less responsible for imagery v.s. real movement + hand v.s. tongue.
     - linear combinations of sites or non-linear (lasso) , MI, CMI
     - do these models explain Figure 2 (spatially)
     - If we train an autoregressive model we can do this for the temporal range before/during stim.)

## Roadmap
0. Create Git
1. Explore/Understand/Convert data
    - Separate signal bands and then do power spectra (Xander, Mandar)  https://neurodsp-tools.github.io/neurodsp/auto_examples/plot_mne_example.html#sphx-glr-auto-examples-plot-mne-example-py
    - Brain channel visualization that matches anatomically (MNE: https://mne.tools/dev/auto_tutorials/epochs/index.html, (https://mne.tools/stable/auto_tutorials/clinical/30_ecog.html#:~:text=MNE%20supports%20working%20with%20more,with%20electrocorticography%20(ECoG)%20data )(Everyone)
    - EDA 
         - inspiration (Mandar)

#### Towards answering some questions
2. GLM analysis as done in fMRI? [ref](https://nilearn-doc-dev.github.io/auto_examples/02_decoding/plot_haxby_glm_decoding.html)
    - y_i = c * X * b_i
    - y_i is the signal recorded from electrode `i` of shape (`t_points`, )
    - X is design matrix of shape (`t_points`, `event_ids`), ideally should be the signal from data gloves
    - c is the contrast vector of shape (, `event_ids`)
    - b_i is the scalar weight for the corresponding electrode
    - contrast `hand` - `tongue` meaning c is [0 -1 1] (??)
    - optimise betas in B, i.e. the weights corresponding to each electrode, showing which electrodes were most active during `hand` events relative to `tongue` events
    - get t-scores corresponding to each beta, showing if and how significant was the difference
    - convert these t-scores to z-scores

3. Decoding:
    - use z-scores from the encoding step to classify between events

#### Maybe goals
3. Why model? There are two possible explanations (recruiting v.s. firing rate incr.)
    - Is there a model that can address this hypothesis? It would address the recurrent relationship.
    - Do we have the spatial or temporal resolution to address this?
    - If we could show : recurrent network

how about predicting when? autoregression models and premotor cortex and planning.
predicting why changes happen? is it learning? priming v. habituation. maybe# nma_motor_imagery
