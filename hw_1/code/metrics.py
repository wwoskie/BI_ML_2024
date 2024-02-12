import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    y_pred = y_pred.astype(float).astype(bool)
    y_true = y_true.astype(float).astype(bool)

    # not a very clever workaround though

    true_pos = sum((y_pred == y_true) & (y_pred == True))
    true_neg = sum((y_pred == y_true) & (y_pred == False))
    false_pos = sum((y_pred != y_true) & (y_pred == True))
    false_neg = sum((y_pred != y_true) & (y_pred == False))

    try:
        precision = true_pos / (true_pos + false_pos)
    except ZeroDivisionError:
        precision = None
        print('No positive values predicted')

    try:
        recall = true_pos / (true_pos + false_neg)
    except ZeroDivisionError:
        recall = None
        print('No true positive and no false negative values predicted')
    
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = None
        print('No true positive values predicted')

    try:
        accuracy = (true_pos + true_neg) / len(y_pred)
    except ZeroDivisionError:
        accuracy = None
        print('Empty y_pred passed')

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    return (sum(y_pred == y_true) / len(y_pred), ) # tuple to fix indexation, sowwy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    try:
        r2 = sum(np.square(y_pred - y_true)) / sum(np.square(y_true - np.mean(y_true)))
    except ZeroDivisionError:
        r2 = None
        print('sum of y_true - y_mean squares is zero')
    return r2,


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    try:
        mse = sum(np.square(y_true - y_pred)) / len(y_true)
    except ZeroDivisionError:
        mse = None
        print('y_true is of length zero')
    return mse,


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    try:
        mae = sum(np.abs(y_true - y_pred)) / len(y_true)
    except ZeroDivisionError:
        mae = None
        print('y_true is of length zero')
    return mae,
    
    