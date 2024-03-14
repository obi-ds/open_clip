import re
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    mean_absolute_error
)



# TODO: Evaluate sign
# TODO: Evaluate number


def parse_t2e_text(text):
    if 'inf' in text:
        return 'inf'
    else:
        digit_text = re.sub('(<end_of_text>|[^0-9+-]|\s)', '', text)
        sign = ''
        digits = ''
        for character in digit_text:
            if character in ['+', '-'] and digits == '':
                sign = character
            elif sign is not None and re.search('\d', character):
                digits += character
            else:
                break
        return sign + digits

def get_parsed_t2e_predictions_labels(model_predictions, model_labels, tokenizer_decode):
    labels = [parse_t2e_text(tokenizer_decode(label)) for label in model_labels]
    predictions = [parse_t2e_text(tokenizer_decode(prediction)) for prediction in model_predictions]
    return labels, predictions


def check_for_integer(prediction):
    # TODO: Maybe change to regex based check?
    try:
        int(prediction)
        return True
    except ValueError:
        return False


def prepare_labels_for_binary_metrics(labels, predictions, time_period):
    if time_period == 0:
        labels = [True if check_for_integer(label) else False for label in labels]
        predictions = [True if check_for_integer(prediction) else False for prediction in predictions]
    elif time_period == -1:
        labels = [int(label) < 0 if check_for_integer(label) else False for label in labels]
        predictions = [int(prediction) < 0 if check_for_integer(prediction) else False for prediction in predictions]
    elif time_period == 1:
        labels = [int(label) >= 0 if check_for_integer(label) else False for label in labels]
        predictions = [int(prediction) >= 0 if check_for_integer(prediction) else False for prediction in predictions]
    elif time_period == 'inf':
        labels = [True if label == 'inf' else False for label in labels]
        predictions = [True if prediction == 'inf' else False for prediction in predictions]
    else:
        raise ValueError('Invalid time period')

    return labels, predictions


def prepare_labels_for_numeric_metrics(labels, predictions, time_period, default_value):
    if time_period == 0:
        labels = [
            label if check_for_integer(prediction) else 0
            for label, prediction in zip(labels, predictions) if check_for_integer(label)
        ]
        predictions = [
            prediction if check_for_integer(prediction) else default_value
            for label, prediction in zip(labels, predictions) if check_for_integer(label)
        ]
    elif time_period == -1:
        labels = [
            label if check_for_integer(prediction) else 0
            for label, prediction in zip(labels, predictions) if check_for_integer(label) and int(label) < 0
        ]
        predictions = [
            prediction if check_for_integer(prediction) else default_value
            for label, prediction in zip(labels, predictions) if check_for_integer(label) and int(label) < 0
        ]

    elif time_period == 1:
        labels = [
            label if check_for_integer(prediction) else 0
            for label, prediction in zip(labels, predictions) if check_for_integer(label) and int(label) >= 0
        ]
        predictions = [
            prediction if check_for_integer(prediction) else default_value
            for label, prediction in zip(labels, predictions) if check_for_integer(label) and int(label) >= 0
        ]

    return labels, predictions


def get_binary_metrics(labels, predictions):
    prediction_eval_types = [('number', 0), ('past_number', -1), ('future_number', 1), ('not_number', 'inf')]
    results = {}
    for prediction_eval_type, time_period in prediction_eval_types:
        metric_labels, metric_predictions = prepare_labels_for_binary_metrics(
            labels=labels, predictions=predictions, time_period=time_period
        )
        (
            results[f'{prediction_eval_type}_precision'],
            results[f'{prediction_eval_type}_recall'],
            results[f'{prediction_eval_type}_f1'],
            results[f'{prediction_eval_type}_number']
        ) = \
        precision_recall_fscore_support(metric_labels, metric_predictions, average='binary')
    return results


def get_numeric_metrics(labels, predictions):
    prediction_eval_types = [('number', 0), ('past_number', -1), ('future_number', 1)]
    results = {}
    for prediction_eval_type, time_period in prediction_eval_types:
        metric_labels, metric_predictions = prepare_labels_for_numeric_metrics(
            labels=labels, predictions=predictions, time_period=time_period
        )
        if metric_labels == []:
            results[f'{prediction_eval_type}_mse'] = -1
            results[f'{prediction_eval_type}_mae'] = -1
            results[f'{prediction_eval_type}_acc'] = 0
        else:
            results[f'{prediction_eval_type}_mse'] = mean_squared_error(metric_labels, metric_predictions)
            results[f'{prediction_eval_type}_mae'] = mean_absolute_error(metric_labels, metric_predictions)
            results[f'{prediction_eval_type}_acc'] = accuracy_score(metric_labels, metric_predictions)

    return results