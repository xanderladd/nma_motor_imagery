from minma.data import get_all_data, get_subject_data, get_epochs, raw_to_signal, get_raw, pickle_dataset
from minma.dsp import epochs_to_PSD_samples, reduce_samples
import os

if __name__ == "__main__":
    # move to root directory
    os.chdir('../')
    event_ids = dict(rest=10, tongue=11, hand=12)
    ECoG_data =  get_all_data()
    n_subjects = len(ECoG_data)
    exps = ['mvmt','imagery']
    for exp_idx, exp in enumerate(exps):
        for sbj_idx in range(n_subjects):
            subject_data = get_subject_data(ECoG_data, sbj_idx, exp_idx)
            # get MNE epochs
            epochs = get_epochs(subject_data, event_ids, load=True)

            ### FULL 3S Decomp  ###
            hfb_samples, lfb_samples, all_samples, labels =  epochs_to_PSD_samples(epochs, tmin=.6, tmax=3)
            ### 1S Decomp  ###
            # hfb_samples, lfb_samples, all_samples, labels  = epoch_windows_to_PSD_samples(epochs, w_size=1, fft_size=256) # window size in seconds
            ###  .5 s Decomp  ###
            # hfb_samples, lfb_samples, all_samples, labels  = epoch_windows_to_PSD_samples(epochs, w_size=.5, fft_size=256) # window size in seconds

            # would change labels
            # labels = np.array([exp + '_' + label for label in labels])

            hfb_integ_psd_samples, hfb_median_psd_samples, hfb_sampled_freqs = reduce_samples(hfb_samples)
            lfb_integ_psd_samples, lfb_median_psd_samples, lfb_sampled_freqs = reduce_samples(lfb_samples)
            all_integ_psd_samples, all_median_psd_samples, all_sampled_freqs = reduce_samples(all_samples)

            pickle_dataset(hfb_integ_psd_samples, hfb_median_psd_samples, hfb_sampled_freqs, labels, path=os.path.join('data',f'sbj_{sbj_idx}'), title=f'{exp}_hfb_3s_700ms_delay')
            pickle_dataset(lfb_integ_psd_samples, lfb_median_psd_samples, lfb_sampled_freqs,  labels, path=os.path.join('data',f'sbj_{sbj_idx}'), title=f'{exp}_lfb_3s_700ms_delay')
            pickle_dataset(all_integ_psd_samples, all_median_psd_samples, all_sampled_freqs, labels, path=os.path.join('data',f'sbj_{sbj_idx}'), title=f'{exp}_all_freq_3s_700ms_delay')