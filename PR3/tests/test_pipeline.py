"""Автотесты пайплайна (запуск: pytest tests/ -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.experiments import build_model, split_train_test, year_folds  # noqa: E402
from src.pipeline.features import TARGET, load_raw, make_dataset, make_pipeline, table_hash  # noqa: E402

INITIAL = Path(__file__).resolve().parents[1] / "data/interim/initial_part.csv"


def test_repeatable_features():
    a, _, _ = make_dataset(INITIAL)
    b, _, _ = make_dataset(INITIAL)
    assert table_hash(a) == table_hash(b)


def test_target_not_in_features():
    X, y, _ = make_dataset(INITIAL)
    assert TARGET not in X.columns and y.notna().sum() > 0


def test_no_future_information():
    """Признаки строк года ≤ T не меняются, если убрать все годы > T."""
    X, _, _ = make_dataset(INITIAL)
    part = load_raw(INITIAL)
    part = part[part["year"] <= 2013]
    Xp, _, _ = make_dataset(part)
    a = X[X["year"] <= 2013].set_index(["city_key", "year"]).sort_index()
    b = Xp.set_index(["city_key", "year"]).sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_time_order_in_folds():
    X, y, _ = make_dataset(INITIAL)
    Xtr, ytr, Xte, _ = split_train_test(X, y)
    for tr, va in year_folds(Xtr):
        assert (Xtr["year"].iloc[tr] + 1).max() < (Xtr["year"].iloc[va] + 1).min()
    assert (Xtr["year"] + 1).max() < (Xte["year"] + 1).min()


def test_unknown_category_does_not_fail():
    X, y, _ = make_dataset(INITIAL)
    Xtr, ytr, _, _ = split_train_test(X, y)
    model = make_pipeline(build_model("logreg_l2"), "C").fit(Xtr, ytr)
    new = Xtr.head(20).copy()
    new["year"] = 2022                     # норматив «с 2021» в обучении не встречался
    p = model.predict_proba(new)
    assert p.shape == (20, 4) and np.allclose(p.sum(axis=1), 1)
