import numpy as np
import pandas as pd

from por_hep import (
    DenseNet201DirectFeatureMap,
    grouped_locked_split_indices,
    stratified_group_holdout_indices,
)
from por_hep.data import load_densenet201_features


def test_loader_reads_1920_features_and_groups_exact_duplicates(tmp_path):
    rng = np.random.RandomState(30)
    X = rng.normal(size=(20, 1920)).astype(np.float32)
    X[10] = X[0]
    y = np.tile([0, 1], 10)
    y[10] = y[0]
    frame = pd.DataFrame(X, columns=["f_%04d" % index for index in range(1920)])
    frame.insert(0, "class_name", np.where(y == 1, "TB", "Normal"))
    frame.insert(0, "label", y)
    frame.insert(0, "image_path", ["image_%d.png" % i for i in range(20)])
    path = tmp_path / "densenet201_por_features.csv"
    frame.to_csv(path, index=False)

    dataset = load_densenet201_features(path)
    assert dataset.X.shape == (20, 1920)
    assert dataset.feature_names[0] == "f_0000"
    assert dataset.feature_names[-1] == "f_1919"
    assert dataset.group_id[0] == dataset.group_id[10]
    assert dataset.grouping_source == "exact_1920_feature_row_hash"


def test_direct_map_removes_only_training_constant_channels():
    rng = np.random.RandomState(31)
    X = rng.normal(size=(80, 12))
    X[:, 3] = 4.0
    X[0, 7] = np.nan
    feature_map = DenseNet201DirectFeatureMap(
        ["f_%04d" % index for index in range(12)]
    ).fit(X)
    transformed = feature_map.transform(X)
    assert transformed.shape == (80, 11)
    assert feature_map.output_feature_names_ == [
        "f_%04d" % index for index in range(12) if index != 3
    ]
    assert np.isfinite(transformed).all()
    assert feature_map.summary()["PCA"] is False


def test_grouped_roles_have_no_shared_group():
    y = np.tile([0, 1], 60)
    groups = np.asarray(["group_%03d" % (index // 2) for index in range(120)])
    # Each two-row group must have one label, not alternating labels.
    y = np.repeat(np.tile([0, 1], 30), 2)
    locked = grouped_locked_split_indices(y, groups, random_state=7)
    work, audit = stratified_group_holdout_indices(
        locked.development, y, groups, random_state=8
    )
    base, operator = stratified_group_holdout_indices(
        work, y, groups, random_state=9
    )
    roles = [base, operator, audit, locked.test]
    for first in range(len(roles)):
        for second in range(first + 1, len(roles)):
            assert not np.intersect1d(
                groups[roles[first]], groups[roles[second]]
            ).size
