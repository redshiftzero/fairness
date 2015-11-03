import numpy as np
import pdb

"""
Metrics for fairness in algorithms
----------------------------------
"""


def disparate_impact(protected_status, model_outcome):
    """
    This metric is based on the legal definition of "disparate impact"
    threshold of 80%. This is determined with respect to a protected class.

    Metric definition:
    ------------------
    If P(Protected=0 | Outcome=1 ) / P(Protected=1 | Outcome=1 ) <= 80%
    then the definition of disparate impact is satisfied.

    Inputs
    ------
    protected_status: numpy ndarray
    model_outcome: numpy ndarray

    Outputs
    -------
    is_discriminatory: bool corresponding to if the outputs demonstrate
                       discrimination
    value_discrim: float between 0-1 describing the degree of discrimination
    """

    if len(protected_status) != len(model_outcome):
        raise ValueError('Input arrays do not have same number of entries')

    # "positive class" are those where predictions = 1
    # "majority class" are those where protected class status = 1
    indices_pos_class, = np.where(protected_status == 1)
    outcomes_pos = model_outcome[indices_pos_class]

    value_discrim = len(np.where(outcomes_pos == 0)) / len(
        np.where(outcomes_pos == 1))

    if value_discrim <= 0.8:
        is_discriminatory = True
    else:
        is_discriminatory = False

    return is_discriminatory, 1 - value_discrim


def group_fairness(protected_status, model_outcome):
    """
    This metric describes how fair the algorithm is with respect
    to the protected class by measuring the statistical parity of the
    two groups - a measure of group fairness.

    Inputs
    ------
    protected_status: numpy ndarray
    model_outcome: numpy ndarray

    Outputs
    -------
    value_discrim: float between 0-1 describing the degree of discrimination
    """

    if len(protected_status) != len(model_outcome):
        raise ValueError('Input arrays do not have same number of entries')

    indices_pos_class, = np.where(protected_status == 1)
    indices_neg_class, = np.where(protected_status == 0)

    outcomes_pos = model_outcome[indices_pos_class]
    outcomes_neg = model_outcome[indices_neg_class]

    value_discrim = np.abs(len(np.where(outcomes_pos == 1)) /
                           len(outcomes_pos) -
                           len(np.where(outcomes_neg == 1)) /
                           len(outcomes_neg))

    return value_discrim


def individual_fairness(feature_df, model_outcome):
    """
    This metric describes how fair the algorithm is with respect
    to the protected class by measuring if similar individuals are classified
    similarly. This is a measure of individual fairness.

    Inputs
    ------
    feature_df: pandas dataframe
    model_outcome: numpy ndarray

    Outputs
    -------
    value_discrim: float between 0-1 describing the degree of discrimination
    """

    if len(feature_df) != len(model_outcome):
        raise ValueError('Input arrays do not have same number of entries')

    pass
    num_items = len(model_outcome)
    k = 3
    sum_nearest_neighbs = 0.
    value_discrim = 1. - 1./(num_items * k) * sum_nearest_neighbs

    return value_discrim


def rel_error_rate(protected_status, model_outcome, truth_outcome):
    """
    This measures the fairness of the error distribution with respect to
    a protected class. It returns the error rates in each protected class
    as well as the relative error rates. Errors that are very unevenly
    distributed is one manifestation of unfairness in algorithms.

    Inputs
    ------
    protected_status: numpy ndarray
    model_outcome: numpy ndarray
    truth_outcome: numpy ndarray

    Outputs
    -------
    error_rate: relative error rates
    """

    if (len(protected_status) != len(model_outcome) or
       len(protected_status) != len(truth_outcome)):
        raise ValueError('Input arrays do not have same number of entries')

    indices_pos_class, = np.where(protected_status == 1)
    indices_neg_class, = np.where(protected_status == 0)

    outcomes_pos_wrong, = np.where(model_outcome[indices_pos_class] -
                                   truth_outcome[indices_pos_class])
    outcomes_neg_wrong, = np.where(model_outcome[indices_neg_class] -
                                   truth_outcome[indices_neg_class])

    pos_error_rate = len(outcomes_pos_wrong) / len(indices_pos_class)
    neg_error_rate = len(outcomes_neg_wrong) / len(indices_neg_class)
    error_rate = np.abs(pos_error_rate - neg_error_rate)

    return error_rate, pos_error_rate, neg_error_rate
