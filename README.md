** one step at a time**

Two questions for candidates round #2:
1. Can we use electrodes to predict ***what*** the stimulus is (hand v. tongue) (real v. imagery) ?
       - 95% classification acc.!!:cook::sunglasses::sunglasses:
2. Can we use the model to show ***how*** electrodes/regions are more/less responsible for imagery v.s. real movement + hand v.s. tongue.
     - linear combinations of sites or non-linear (lasso) , MI, CMI
     - do these models explain Figure 2 (spatially)
     - If we train an autoregressive model we can do this for the temporal range before/during stim.

Roadmap
0. Create Git
1. Analyze data
    - Separate signal bands and then do power spectra (Xander, Mandar)  https://neurodsp-tools.github.io/neurodsp/auto_examples/plot_mne_example.html#sphx-glr-auto-examples-plot-mne-example-py
    - Brain channel visualization that matches anatomically (MNE: https://mne.tools/dev/auto_tutorials/epochs/index.html, (https://mne.tools/stable/auto_tutorials/clinical/30_ecog.html#:~:text=MNE%20supports%20working%20with%20more,with%20electrocorticography%20(ECoG)%20data )(Everyone)
    - EDA 
         - inspiration (Mandar)

**careful not to try _everything_ lets pick our battles esp. considering we have two weeks **
---------------------------------------------------------------------------------------------------------------------
3. Why model? There are two possible explanations (recruiting v.s. firing rate incr.)
    - Is there a model that can address this hypothesis? It would address the recurrent relationship.
    - Do we have the spatial or temporal resolution to address this?
    - If we could show : recurrent network

how about predicting when? autoregression models and premotor cortex and planning.
predicting why changes happen? is it learning? priming v. habituation. maybe# nma_motor_imagery
