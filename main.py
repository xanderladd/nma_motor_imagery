import os
from load_data import load_psd_dataset, subset_data_paths



if __name__ == "__main__":
    
    base_path = os.path.join('data','sbj_0')
    title = 'mvmt_hfb_3s'
    curr_data = load_psd_dataset(base_path,title)
    # no need to do this til we look at imagery
    # curr_data['labels'] = update_labels(curr_data['labels'],'mvmt')
    
    import pdb; pdb.set_trace()
    # os.makedirs('plots', exist_ok=True)
