"""Locked evaluation splits and auditable training-role sampling."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass(frozen=True)
class LockedSplitIndices:
    development: np.ndarray
    test: np.ndarray
    protocol: str


@dataclass(frozen=True)
class RoleIndices:
    base: np.ndarray
    operator_validation: np.ndarray
    dimension_audit: np.ndarray


def locked_split_indices(y, protocol="random", test_size=0.20,
                         random_state=42, official_test_rows=500000):
    """Return indices without materializing copies of the feature matrix.

    ``uci-official`` preserves the documented SUSY ordering: the first
    4,500,000 examples are development data and the final 500,000 are test.
    The caller is responsible for requiring the canonical five-million-row
    file before claiming the official benchmark protocol.
    """
    y = np.asarray(y, dtype=int)
    indices = np.arange(len(y), dtype=int)
    if protocol == "uci-official":
        official_test_rows = int(official_test_rows)
        if official_test_rows <= 0 or len(y) <= official_test_rows:
            raise ValueError("Not enough rows for the official SUSY test tail.")
        boundary = len(y) - official_test_rows
        return LockedSplitIndices(
            development=indices[:boundary],
            test=indices[boundary:],
            protocol="uci-official-last-500000",
        )
    if protocol != "random":
        raise ValueError("protocol must be random or uci-official.")
    development, test = train_test_split(
        indices,
        test_size=float(test_size),
        stratify=y,
        random_state=int(random_state),
    )
    return LockedSplitIndices(
        development=np.asarray(development, dtype=int),
        test=np.asarray(test, dtype=int),
        protocol="stratified-random",
    )


def stratified_group_holdout_indices(
    indices, y, groups, test_size=0.20, random_state=42
):
    """Approximate a stratified holdout while keeping groups indivisible."""
    indices = np.asarray(indices, dtype=int)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("y and groups must have equal length.")
    fraction = float(test_size)
    if not 0.05 <= fraction <= 0.50:
        raise ValueError("group holdout fraction must lie in [0.05, 0.50].")
    local_y = y[indices]
    local_groups = groups[indices]
    n_splits = max(2, int(round(1.0 / fraction)))
    group_frame = {}
    for group, label in zip(local_groups, local_y):
        group_frame.setdefault(str(group), set()).add(int(label))
    if any(len(labels) != 1 for labels in group_frame.values()):
        raise ValueError("A split group contains conflicting labels.")
    counts = np.bincount(
        [next(iter(labels)) for labels in group_frame.values()], minlength=2
    )
    if np.min(counts) < n_splits:
        raise ValueError(
            "Too few independent groups per class for %d-fold group splitting."
            % n_splits
        )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=int(random_state)
    )
    prevalence = float(np.mean(local_y))
    candidates = []
    dummy = np.zeros((len(indices), 1), dtype=float)
    for train_local, test_local in splitter.split(
        dummy, local_y, groups=local_groups
    ):
        size_error = abs(len(test_local) / len(indices) - fraction)
        balance_error = abs(float(np.mean(local_y[test_local])) - prevalence)
        candidates.append((size_error + balance_error, train_local, test_local))
    _, train_local, test_local = min(candidates, key=lambda item: item[0])
    train = indices[np.asarray(train_local, dtype=int)]
    test = indices[np.asarray(test_local, dtype=int)]
    if np.intersect1d(groups[train], groups[test]).size:
        raise RuntimeError("Group leakage detected after grouped split.")
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def grouped_locked_split_indices(
    y, groups, test_size=0.20, random_state=42
):
    indices = np.arange(len(y), dtype=int)
    development, test = stratified_group_holdout_indices(
        indices,
        y,
        groups,
        test_size=test_size,
        random_state=random_state,
    )
    return LockedSplitIndices(
        development=development,
        test=test,
        protocol="stratified-group-random",
    )


def _stratified_take(indices, y, size, random_state):
    indices = np.asarray(indices, dtype=int)
    size = int(size)
    if size <= 0 or size > len(indices):
        raise ValueError("Requested role size is outside the available range.")
    if size == len(indices):
        return indices.copy(), np.empty(0, dtype=int)
    take, remain = train_test_split(
        indices,
        train_size=size,
        stratify=np.asarray(y, dtype=int)[indices],
        random_state=int(random_state),
    )
    return np.asarray(take, dtype=int), np.asarray(remain, dtype=int)


def fixed_size_role_indices(development_indices, y, base_size,
                            operator_size, dimension_audit_size,
                            random_state=42):
    """Draw three mutually disjoint stratified development roles."""
    development_indices = np.asarray(development_indices, dtype=int)
    requested = int(base_size) + int(operator_size) + int(dimension_audit_size)
    if min(int(base_size), int(operator_size), int(dimension_audit_size)) <= 0:
        raise ValueError("All fixed role sizes must be positive.")
    if requested > len(development_indices):
        raise ValueError("Fixed development roles exceed available rows.")
    selected = development_indices
    if requested < len(development_indices):
        selected, _ = _stratified_take(
            development_indices, y, requested, random_state
        )
    base, remain = _stratified_take(selected, y, base_size, random_state + 1)
    operator, remain = _stratified_take(
        remain, y, operator_size, random_state + 2
    )
    if len(remain) != int(dimension_audit_size):
        audit, _ = _stratified_take(
            remain, y, dimension_audit_size, random_state + 3
        )
    else:
        audit = remain
    return RoleIndices(base, operator, np.asarray(audit, dtype=int))


def stratified_refit_indices(development_indices, y, size=0,
                             random_state=42):
    """Select the post-selection refit pool; zero means all development rows."""
    development_indices = np.asarray(development_indices, dtype=int)
    if int(size) <= 0 or int(size) >= len(development_indices):
        return development_indices.copy()
    selected, _ = _stratified_take(
        development_indices, y, int(size), random_state
    )
    return selected
