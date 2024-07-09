import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def get_past_and_future_ranges(binned_data_columns):
    range_columns = binned_data_columns[binned_data_columns.str.contains('\(')]
    mask = range_columns.str.contains('-\d+]')
    past_ranges = range_columns[mask]
    future_ranges = range_columns[~mask]
    return past_ranges, future_ranges

def get_counts_in_range(binned_data, ranges):
    return binned_data.loc[:, ranges].sum(axis=1)

def extract_positives_range(binned_data, ranges):
    counts = get_counts_in_range(binned_data=binned_data, ranges=ranges)
    return counts >= 1

def extract_positives_range_no_past(binned_data, ranges, past_ranges):
    counts = get_counts_in_range(binned_data=binned_data, ranges=ranges)
    past_counts = get_counts_in_range(binned_data=binned_data, ranges=past_ranges)
    return (counts >= 1) & (past_counts == 0)

def extract_positives_range_yes_past(binned_data, ranges, past_ranges):
    counts = get_counts_in_range(binned_data=binned_data, ranges=ranges)
    past_counts = get_counts_in_range(binned_data=binned_data, ranges=past_ranges)
    return (counts >= 1) & (past_counts >= 1)

def extract_positives_sub_range_yes_past(binned_data, ranges, past_ranges, sub_range):
    counts = get_counts_in_range(binned_data=binned_data, ranges=ranges)
    past_counts = get_counts_in_range(binned_data=binned_data, ranges=past_ranges)
    sub_counts = get_counts_in_range(binned_data=binned_data, ranges=sub_range)
    return (counts >= 1) & (past_counts >= 1) & (sub_counts == 0)

def extract_negatives_range(binned_data, ranges):
    counts = get_counts_in_range(binned_data=binned_data, ranges=ranges)
    return counts == 0


def get_positives_for_range(binned_data, ranges, past_ranges):
    positives_range = extract_positives_range(
        binned_data=binned_data, ranges=ranges
    )

    positives_range_no_past = extract_positives_range_no_past(
        binned_data=binned_data, ranges=ranges, past_ranges=past_ranges
    )

    positives_range_yes_past = extract_positives_range_yes_past(
        binned_data=binned_data, ranges=ranges, past_ranges=past_ranges
    )

    return positives_range, positives_range_no_past, positives_range_yes_past

def get_negatives_for_range(binned_data, ranges, past_ranges, future_ranges):
    negatives_range = extract_negatives_range(
        binned_data=binned_data, ranges=past_ranges + ranges + future_ranges
    )

    negatives_range_no_past = extract_negatives_range(
        binned_data=binned_data, ranges=past_ranges + ranges
    )

    negatives_range_yes_past = extract_negatives_range(
        binned_data=binned_data, ranges=ranges
    )

    return negatives_range, negatives_range_no_past, negatives_range_yes_past

def add_positives(binned_data, prediction_range, positives_range, positives_range_no_past, positives_range_yes_past):
    binned_data[f'{prediction_range} pos'] = positives_range
    binned_data[f'{prediction_range} pos-'] = positives_range_no_past
    binned_data[f'{prediction_range} pos+'] = positives_range_yes_past

    return binned_data

def add_negatives(binned_data, prediction_range, negatives_range, negatives_range_no_past, negatives_range_yes_past):
    binned_data[f'{prediction_range} neg'] = negatives_range
    binned_data[f'{prediction_range} neg-'] = negatives_range_no_past
    binned_data[f'{prediction_range} neg+'] = negatives_range_yes_past

    return binned_data

def compute_and_add_positives(binned_data, prediction_range, ranges, past_ranges):
    positives_range, positives_range_no_past, positives_range_yes_past = get_positives_for_range(
        binned_data=binned_data,
        ranges=ranges,
        past_ranges=past_ranges
    )

    binned_data = add_positives(
        binned_data=binned_data,
        prediction_range=prediction_range,
        positives_range=positives_range,
        positives_range_no_past=positives_range_no_past,
        positives_range_yes_past=positives_range_yes_past
    )

    return binned_data

def compute_and_add_sub_positives(binned_data, prediction_range, ranges, past_ranges):
    past_ranges_extended = list(past_ranges) + list(ranges)

    positives_range, positives_range_no_past, _ = get_positives_for_range(
        binned_data=binned_data,
        ranges=[prediction_range],
        past_ranges=past_ranges_extended
    )

    positives_range_yes_past = extract_positives_sub_range_yes_past(
        binned_data=binned_data, ranges=[prediction_range], past_ranges=past_ranges, sub_range=ranges
    )

    binned_data = add_positives(
        binned_data=binned_data,
        prediction_range=prediction_range,
        positives_range=positives_range,
        positives_range_no_past=positives_range_no_past,
        positives_range_yes_past=positives_range_yes_past
    )

    return binned_data

def compute_and_add_negatives(binned_data, prediction_range, ranges, past_ranges, future_ranges):

    negatives_range, negatives_range_no_past, negatives_range_yes_past = get_negatives_for_range(
        binned_data=binned_data,
        past_ranges=past_ranges,
        ranges=ranges,
        future_ranges=future_ranges
    )

    binned_data = add_negatives(
        binned_data=binned_data,
        prediction_range=prediction_range,
        negatives_range=negatives_range,
        negatives_range_no_past=negatives_range_no_past,
        negatives_range_yes_past=negatives_range_yes_past
    )

    return binned_data

def get_compare_columns(binned_data_columns):
    positive_columns = binned_data_columns[binned_data_columns.str.contains('pos')]
    negative_columns = binned_data_columns[binned_data_columns.str.contains('neg')]
    compare_columns = [
        (positive_column, negative_column) for positive_column in positive_columns for negative_column in negative_columns
    ]
    return compare_columns


def get_metrics(eval_df, phecode, positive_column, negative_column):
    label_type = positive_column.split()[-1] + '/' + negative_column.split()[-1]
    prediction_range = ' '.join(positive_column.split()[:-1])

    positives = eval_df[eval_df[positive_column]]
    negatives = eval_df[eval_df[negative_column]]

    positive_counts = len(positives)
    negative_counts = len(negatives)

    if positives.empty or negatives.empty:
        y_auc = None
        p_auc = None
    else:
        y_scores = pd.concat([positives['yes'], negatives['yes']])
        p_scores = pd.concat(
            [
                np.exp(positives['yes']) / (np.exp(positives['yes']) + np.exp(positives['no'])),
                np.exp(negatives['yes']) / (np.exp(negatives['yes']) + np.exp(negatives['no']))
            ]
        )
        labels = [1] * positive_counts + [0] * negative_counts
        y_auc = roc_auc_score(labels, y_scores)
        p_auc = roc_auc_score(labels, p_scores)
        # TODO: Add auprc, f1,


    return {
        'phecode': phecode,
        'label_type': label_type,
        'prediction_range': prediction_range,
        'positives_count': positive_counts,
        'negatives_count': negative_counts,
        'auc': y_auc,
        'p_auc': p_auc
    }
