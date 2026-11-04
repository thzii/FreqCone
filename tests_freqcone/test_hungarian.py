"""The validation metric uses the same Hungarian accuracy as the test protocol's 'All'."""
import numpy as np

from project_utils.cluster_and_log_utils import split_cluster_acc_v2
from project_utils.cluster_utils import cluster_acc


def test_cluster_acc_equals_split_cluster_acc_v2_all():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 30, 400)
        pred = np.where(rng.random(400) < 0.6, (y * 7 + 3) % 30, rng.integers(0, 30, 400))
        mask = y < 15
        assert abs(cluster_acc(y, pred) - split_cluster_acc_v2(y, pred, mask)[0]) < 1e-12


def test_metric_ignores_cluster_names():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 10, 200)
    pred = np.where(rng.random(200) < 0.8, y, rng.integers(0, 10, 200))
    perm = rng.permutation(10)
    assert abs(cluster_acc(y, pred) - cluster_acc(y, perm[pred])) < 1e-12
