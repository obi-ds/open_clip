import seaborn as sns
import matplotlib.pyplot as plt

def get_counts_df(dataframe, prompt_range, label_type, threshold):
    count_filter = (
        (dataframe['label_type'] == label_type)
        & (dataframe['positives_count'] >= threshold)
        & (dataframe['prediction_range'].isin(prompt_range))
    )
    count_filter_codes = dataframe['phecode'].isin(dataframe[count_filter]['phecode'])
    return dataframe[count_filter_codes]

def get_auc_dfs(dataframe, prompt_range, label_type, threshold):
    high_filter = (
            (dataframe['label_type'] == label_type) &
            (dataframe['auc'] >= threshold) &
            (dataframe['prediction_range'].isin(prompt_range))
    )
    auc_filter_codes = dataframe['phecode'].isin(dataframe[high_filter]['phecode'])

    auc_high_df = dataframe[auc_filter_codes]
    auc_low_df = dataframe[~auc_filter_codes]
    return auc_high_df, auc_low_df

def get_auc_sub_dfs(dataframe, label_type, prompt_range, threshold):
    high_filter = (
            (dataframe['label_type'] == label_type) &
            ((dataframe['auc'] >= threshold) | (dataframe['auc'].isna())) &
            (dataframe['prediction_range'].isin(prompt_range))
    )
    auc_filter_codes = dataframe['phecode'].isin(dataframe[high_filter]['phecode'])
    return dataframe[high_filter]

def plot_auc_and_counts(dataframe):
    dataframe_auc_vis = dataframe.pivot(index=['phecode'], columns='prediction_range', values=['auc'])
    dataframe_count_vis = dataframe.pivot(index=['phecode'], columns='prediction_range', values=['positives_count'])
    fig, (ax,ax2) = plt.subplots(ncols=2, squeeze=True, figsize=(13, 13))
    fig.subplots_adjust(wspace=0.5)
    sns.heatmap(dataframe_auc_vis, annot=True, ax=ax, annot_kws={'fontsize':5})
    # fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
    sns.heatmap(dataframe_count_vis, ax=ax2, annot=True, annot_kws={'fontsize':5}, fmt='.0f')
    # fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
    #ax2.yaxis.tick_right()
    #ax2.tick_params(rotation=0)
    plt.show()

def get_poor_performing_codes(dataframe, topk, ascending):
    return dataframe.groupby('phecode')['auc'].mean().sort_values(ascending=ascending)[:topk]