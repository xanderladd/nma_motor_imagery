import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import os
# todo move this out to some plotting file?
def plot_conf_mats(conf_mats, model, lbl_corr, title, prefix, path=None):
    conf_mat = np.sum(np.stack(conf_mats), axis=0)
    df_cm = pd.DataFrame(conf_mat, index = [lbl_corr[key] for key in lbl_corr.keys()],
                    columns =  [lbl_corr[key] for key in lbl_corr.keys()])
    fig  = plt.figure(figsize = (10,7))
    ax = fig.gca()
    sns.heatmap(df_cm, annot=True, ax=ax, fmt='g', cmap='Reds')
    ax.set_ylabel('true label')
    ax.set_xlabel('predicted label')
    plt.title(title)
    if not path:
        fig.savefig(f'{prefix}/plots/{model}_conf.png')
    else:
        os.makedirs(os.path.join(prefix,f'plots',path), exist_ok=True)
        fig.savefig(os.path.join(prefix, f'plots',path,f'{model}_conf.png'))
    plt.close(fig)

def postprocess_classif_metrics(results, lbl_corr, prefix, title=''):
    conf_mats = {}
    df = pd.DataFrame(columns=['model', 'subject','precison','recall','F1', 'acc'])
    if not 'outputs' in prefix:
        prefix = os.path.join('outputs',prefix)
    os.makedirs(prefix, exist_ok=True)

    for sbj_key in results.keys():
        y, y_pred = results[sbj_key]['y'], results[sbj_key]['y_pred']
        model = results[sbj_key]['model']
        # remove subject_extensions since we have models as a column now.
        sbj_key = '_'.join(sbj_key.split('_')[:2])
        conf_mat = confusion_matrix(y, y_pred)
        plot_conf_mats([conf_mat], model, lbl_corr, title, prefix, path=sbj_key)
        if model not in conf_mats:
            conf_mats[model] = [conf_mat]
        else:
            conf_mats[model].append(conf_mat)

        if len(np.unique(y)) > 2: #multi class
            row = {'model': model, 'subject': sbj_key, 'precison': precision_score(y, y_pred,average='weighted'), \
            'recall': recall_score(y, y_pred, average='weighted')  , \
            'F1': f1_score(y, y_pred, average='weighted'),\
             'acc': accuracy_score(y, y_pred) }
        else: # binary cls
            row = {'model': model, 'subject': sbj_key, 'precison': precision_score(y, y_pred), \
            'recall': recall_score(y, y_pred)  , 'F1': f1_score(y, y_pred), \
            'acc': accuracy_score(y, y_pred) }
        df = df.append(row, ignore_index=True)
    # subject_expanded = df['subject'].str.split('_', expand=True)
    # df['subject'] = subject_expanded[0] + '_' + subject_expanded[1]
    barplot_scores(df, prefix, col='acc')
    barplot_scores(df, prefix, col='recall')
    barplot_scores(df, prefix, col='F1')
    for model in df['model'].unique():
        plot_conf_mats(conf_mats[model], model, lbl_corr, title, prefix)
    sbj_agg_df = df.groupby(['model','subject']).agg([np.mean, np.std])
    sbj_agg_df.columns = ['_'.join((col[0], str(col[1]))) for col in sbj_agg_df.columns]

    agg_df = df.groupby(['model']).agg([np.mean, np.std])
    agg_df.columns = ['_'.join((col[0], str(col[1]))) for col in agg_df.columns]
    print(agg_df)
    agg_df.to_csv(os.path.join(prefix,'agg_results_df.csv'))
    sbj_agg_df.to_csv(os.path.join(prefix,'subj_results_df.csv'))


def barplot_scores(df, prefix, col):
    fig = plt.figure()
    ax = fig.gca()
    sns.barplot(x='subject', y=col, hue='model', data=df, ax=ax)
    plt.title(col)
    fig.savefig(os.path.join(prefix, 'plots',col+'_scores.png'))