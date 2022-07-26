

import os
from load_data import load_psd_dataset, subset_data_paths, append_dataset, update_labels
from decoder_model import combine_datasets,filter_label
import seaborn as sns
import matplotlib.pyplot as plt
if __name__ == "__main__":
    
    # for all six subjects

    for i in range(7):

        curr_data =  combine_datasets(subjects=[i], file_keys=['mvmt|imagery','3s','hfb'])
        #filtering out the rest
        curr_data = filter_label(curr_data,'rest')
        X = curr_data['integrated_psd']
        plt.figure()
        psd_feature_sns_plot = sns.heatmap(X)
        resultant_figure_path = './plots/int_psd_feat_heatmap_sub%i' %i
        plt.savefig(resultant_figure_path,dpi=400)



# %%
