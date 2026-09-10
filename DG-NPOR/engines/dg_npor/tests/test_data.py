import numpy as np
import pandas as pd

from por_hep.data import load_susy


def test_susy_loader_reads_label_and_18_features(tmp_path):
    rng = np.random.RandomState(31)
    y = np.tile([0, 1], 20)
    frame = pd.DataFrame(np.column_stack([y, rng.normal(size=(40, 18))]))
    path = tmp_path / "susy.csv"
    frame.to_csv(path, header=False, index=False)
    dataset = load_susy(path, sample=30)
    assert dataset.X.shape == (30, 18)
    assert dataset.y.shape == (30,)
    assert len(dataset.feature_names) == 18
    assert dataset.X.dtype == np.float32
