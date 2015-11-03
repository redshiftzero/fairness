import pandas as pd
import numpy as np
import pdb
from nose.tools import assert_equals

from fairness import metrics


class TestMetricsFunctions:
    def test_disparate_impact(self):
        data = pd.read_csv('tests/validation.csv', index_col=0)

        protected_status = data['gender'].values
        model_outcome = data['prediction'].values

        assert_equals(metrics.disparate_impact(protected_status,
                                               model_outcome), (False, 0.0))

    def test_group_fairness(self):
        data = pd.read_csv('tests/validation.csv', index_col=0)

        protected_status = data['gender'].values
        model_outcome = data['prediction'].values

        assert_equals(metrics.group_fairness(protected_status,
                                             model_outcome), 0.0)

    def test_individual_fairness(self):
        data = pd.read_csv('tests/validation.csv', index_col=0)

        model_outcome = data['prediction'].values
        pass
        assert_equals(metrics.individual_fairness(data,
                                                  model_outcome), 1.0)

    def test_error_rate(self):
        data = pd.read_csv('tests/validation.csv', index_col=0)

        model_outcome = data['prediction'].values
        truth_outcome = data['truth'].values
        protected_class = data['gender'].values

        assert_equals(metrics.rel_error_rate(protected_class,
                                             model_outcome,
                                             truth_outcome),
                      (0.125, 0.625, 0.75))
