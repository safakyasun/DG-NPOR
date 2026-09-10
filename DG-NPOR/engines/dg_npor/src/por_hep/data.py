"""Dataset loading for particle tables and frozen DenseNet201 features."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .susy_features import SUSY_FEATURE_NAMES


DROP_COLUMNS = {
    "EventId", "EventID", "event_id", "Weight", "weight",
    "KaggleWeight", "KaggleSet",
}


@dataclass(frozen=True)
class ParticleDataset:
    X: np.ndarray
    y: np.ndarray
    row_index: np.ndarray
    feature_names: list
    dataset_name: str
    group_id: np.ndarray = None
    grouping_source: str = "none"


def _encode_labels(series):
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({
        "b": 0,
        "s": 1,
        "background": 0,
        "signal": 1,
        "0": 0,
        "1": 1,
        "0.0": 0,
        "1.0": 1,
    })
    if mapped.isna().any():
        unique = sorted(series.dropna().unique().tolist())
        if len(unique) != 2:
            raise ValueError("Expected a binary label column; got %s." % unique)
        mapped = series.map({unique[0]: 0, unique[1]: 1})
    return mapped.to_numpy(dtype=int)


def _stratified_limit(X, y, indices, sample, random_state):
    if int(sample) <= 0 or len(X) <= int(sample):
        return X, y, indices
    X_keep, _, y_keep, _, index_keep, _ = train_test_split(
        X,
        y,
        indices,
        train_size=int(sample),
        stratify=y,
        random_state=int(random_state),
    )
    return X_keep, y_keep, index_keep


def load_atlas_higgs(path, sample=20000, random_state=42):
    frame = pd.read_csv(path)
    target = "Label" if "Label" in frame.columns else "label"
    if target not in frame.columns:
        raise ValueError("ATLAS Higgs CSV must contain Label or label.")
    y = _encode_labels(frame[target])
    features = frame.drop(
        columns=[target] + [name for name in DROP_COLUMNS if name in frame.columns],
        errors="ignore",
    ).apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    X = features.to_numpy(dtype=float)
    indices = np.arange(len(X), dtype=int)
    X, y, indices = _stratified_limit(X, y, indices, sample, random_state)
    return ParticleDataset(
        X=X,
        y=y,
        row_index=indices,
        feature_names=features.columns.tolist(),
        dataset_name="ATLAS Higgs Challenge 2014",
        group_id=np.asarray(indices, dtype=str),
        grouping_source="unique_source_row",
    )


def load_susy(path, sample=20000, random_state=42):
    del random_state  # SUSY sampling deliberately preserves the documented row order.
    nrows = None if int(sample) <= 0 else int(sample)
    frame = pd.read_csv(
        path,
        header=None,
        nrows=nrows,
        compression="infer",
        dtype=np.float32,
    )
    if frame.shape[1] != 19:
        raise ValueError(
            "UCI SUSY requires label + 18 features (19 columns); found %d."
            % frame.shape[1]
        )
    y = frame.iloc[:, 0].to_numpy(dtype=int)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("UCI SUSY must contain labels 0 and 1.")
    # Five million SUSY rows fit comfortably only if the raw table is not
    # needlessly promoted to float64.  Downstream numerical routines promote
    # the much smaller role-specific samples when required.
    X = frame.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
    return ParticleDataset(
        X=X,
        y=y,
        row_index=np.arange(len(X), dtype=int),
        feature_names=list(SUSY_FEATURE_NAMES),
        dataset_name="UCI SUSY",
        group_id=np.arange(len(X), dtype=int).astype(str),
        grouping_source="unique_source_row",
    )


def _feature_hash_groups(features):
    """Return deterministic exact-row groups without using labels."""
    hashed = pd.util.hash_pandas_object(features, index=False)
    return hashed.astype(str).to_numpy()


def _group_consistency(groups, y):
    audit = pd.DataFrame({"group": groups, "label": y})
    conflicting = audit.groupby("group", sort=False)["label"].nunique()
    if (conflicting > 1).any():
        raise ValueError(
            "At least one DenseNet group contains conflicting class labels."
        )


def _group_preserving_limit(X, y, indices, groups, sample, random_state):
    if int(sample) <= 0 or len(X) <= int(sample):
        return X, y, indices, groups
    unique_groups, first = np.unique(groups, return_index=True)
    group_labels = y[first]
    target_groups = int(round(len(unique_groups) * int(sample) / len(X)))
    target_groups = min(max(target_groups, 2), len(unique_groups) - 1)
    selected_groups, _ = train_test_split(
        unique_groups,
        train_size=target_groups,
        stratify=group_labels,
        random_state=int(random_state),
    )
    keep = np.isin(groups, selected_groups)
    return X[keep], y[keep], indices[keep], groups[keep]


def load_densenet201_features(
    path, sample=0, random_state=42, group_column=None
):
    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise ValueError("DenseNet feature CSV must contain a label column.")
    feature_names = sorted(
        [
            name for name in frame.columns
            if name.startswith("f_") and name[2:].isdigit()
        ],
        key=lambda name: int(name[2:]),
    )
    if len(feature_names) != 1920:
        raise ValueError(
            "Expected DenseNet201 GAP columns f_0000...f_1919; found %d."
            % len(feature_names)
        )
    features = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    X = features.to_numpy(dtype=np.float32)
    y = _encode_labels(frame["label"])
    indices = np.arange(len(frame), dtype=int)

    if group_column is not None:
        if group_column not in frame.columns:
            raise ValueError("group-column %r is absent from the CSV." % group_column)
        if frame[group_column].isna().any():
            raise ValueError("group-column contains missing values.")
        groups = frame[group_column].astype(str).to_numpy()
        grouping_source = "explicit_column:%s" % group_column
    else:
        groups = _feature_hash_groups(features)
        grouping_source = "exact_1920_feature_row_hash"
    _group_consistency(groups, y)
    X, y, indices, groups = _group_preserving_limit(
        X, y, indices, groups, sample, random_state
    )
    return ParticleDataset(
        X=X,
        y=y,
        row_index=indices,
        feature_names=feature_names,
        dataset_name="DenseNet201 frozen GAP lung-image features",
        group_id=np.asarray(groups),
        grouping_source=grouping_source,
    )


def load_particle_dataset(
    path, dataset, sample=20000, random_state=42, group_column=None
):
    if dataset == "atlas-higgs":
        return load_atlas_higgs(path, sample=sample, random_state=random_state)
    if dataset == "susy":
        return load_susy(path, sample=sample, random_state=random_state)
    if dataset == "densenet201":
        return load_densenet201_features(
            path,
            sample=sample,
            random_state=random_state,
            group_column=group_column,
        )
    raise ValueError("dataset must be atlas-higgs, susy, or densenet201.")
