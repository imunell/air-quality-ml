"""Схема экспериментов практической работы № 3: разбиение, кросс-валидация по годам, метрики."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (average_precision_score, cohen_kappa_score, f1_score, recall_score,
                             roc_auc_score)

from src.pipeline.features import HIGH_FROM, LEVELS, make_pipeline

SEED = 42
TEST_FIRST_TARGET_YEAR = 2016      # тест: прогнозы на 2016–2017 годы (признаки за 2015 и 2016)
CV_VAL_YEARS = [2011, 2012, 2013, 2014, 2015]   # годы, на которые строится прогноз в проверочных блоках
THRESHOLD = 0.5                    # стандартный порог для Recall (оптимальный порог — в ПР № 4)


# ---------------------------------------------------------------- разбиение
def split_train_test(X: pd.DataFrame, y: pd.Series):
    """Временное разбиение: обучение — прогнозы на 2008–2015 годы, тест — на 2016–2017.
    Строки без известной цели (последний год, уровень −9) исключаются."""
    ok = y.notna().to_numpy()
    ty = (X["year"] + 1).to_numpy()
    tr = ok & (ty < TEST_FIRST_TARGET_YEAR)
    te = ok & (ty >= TEST_FIRST_TARGET_YEAR)
    return (X[tr].reset_index(drop=True), y[tr].astype(int).reset_index(drop=True),
            X[te].reset_index(drop=True), y[te].astype(int).reset_index(drop=True))


def year_folds(X_train: pd.DataFrame, val_years=CV_VAL_YEARS):
    """Расширяющееся окно по годам: обучение на прогнозах за годы < v, проверка на году v."""
    ty = (X_train["year"] + 1).to_numpy()
    return [(np.where(ty < v)[0], np.where(ty == v)[0]) for v in val_years]


# ---------------------------------------------------------------- метрики
def p_high(proba: np.ndarray, classes) -> np.ndarray:
    classes = list(classes)
    idx = [i for i, c in enumerate(classes) if c >= HIGH_FROM]
    return proba[:, idx].sum(axis=1)


def full_proba(est, X) -> np.ndarray:
    """Вероятности в порядке уровней 1..4 (если в обучении не было какого-то уровня — нули)."""
    p = est.predict_proba(X)
    out = np.zeros((len(X), len(LEVELS)))
    for j, c in enumerate(est.classes_):
        out[:, LEVELS.index(int(c))] = p[:, j]
    return out


def metrics(y_true, proba: np.ndarray) -> dict:
    """ROC-AUC и PR-AUC для «уровень 3–4», macro-F1 и QWK по четырём уровням, Recall (3–4) при пороге 0.5."""
    y_true = np.asarray(y_true)
    yh = (y_true >= HIGH_FROM).astype(int)
    ph = p_high(proba, LEVELS)
    pred = np.array(LEVELS)[proba.argmax(axis=1)]
    return {"roc_auc": roc_auc_score(yh, ph),
            "pr_auc": average_precision_score(yh, ph),
            "macro_f1": f1_score(y_true, pred, average="macro", labels=LEVELS),
            "qwk": cohen_kappa_score(y_true, pred, weights="quadratic", labels=LEVELS),
            "recall_high": recall_score(yh, (ph >= THRESHOLD).astype(int), zero_division=0)}


def high_auc_scorer(est, X, y):
    """Скорер для GridSearchCV: ROC-AUC по вероятности уровней 3–4."""
    return roc_auc_score((np.asarray(y) >= HIGH_FROM).astype(int), p_high(full_proba(est, X), LEVELS))


# ---------------------------------------------------------------- базовые решения
class PersistenceClassifier(BaseEstimator, ClassifierMixin):
    """Правило предметной области: уровень следующего года равен уровню текущего."""

    def __init__(self, column="level_lag1"):
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.array(LEVELS)
        self.mode_ = int(pd.Series(y).mode().iloc[0])
        return self

    def predict_proba(self, X):
        lvl = pd.to_numeric(X[self.column], errors="coerce").fillna(self.mode_).round().clip(1, 4).astype(int).to_numpy()
        p = np.full((len(X), 4), 0.05)
        p[np.arange(len(X)), lvl - 1] = 0.85
        return p

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------- кросс-валидация
def cv_evaluate(model, X, y, folds, feature_set="C", name=None) -> dict:
    """Кросс-валидация по годам. model — классификатор без предобработки (Pipeline собирается здесь)
    или готовый объект с ключом 'raw' (для правила persistence)."""
    rows, fit_t = [], []
    for tr, va in folds:
        est = clone(model) if getattr(model, "_raw", False) else make_pipeline(clone(model), feature_set)
        t0 = time.perf_counter(); est.fit(X.iloc[tr], y.iloc[tr]); fit_t.append(time.perf_counter() - t0)
        rows.append(metrics(y.iloc[va], full_proba(est, X.iloc[va])))
    df = pd.DataFrame(rows)
    out = {"model": name, "n_folds": len(folds), "fit_time_s": float(np.mean(fit_t))}
    for c in df:
        out[f"{c}_mean"], out[f"{c}_std"] = float(df[c].mean()), float(df[c].std(ddof=1))
    out["roc_auc_folds"] = [round(float(v), 4) for v in df["roc_auc"]]
    return out


def mark_raw(est):
    est._raw = True
    return est


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


# ---------------------------------------------------------------- реестр моделей
def build_model(name: str, **params):
    """Фабрика моделей: имя + параметры (так итоговая модель записывается в json и воспроизводится)."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    reg = {
        "dummy": lambda p: DummyClassifier(strategy="prior"),
        "logreg_l2": lambda p: LogisticRegression(**{"C": 1.0, "max_iter": 3000, **p}),
        "logreg_l1": lambda p: LogisticRegression(**{"penalty": "l1", "solver": "saga", "C": 0.1,
                                                     "max_iter": 5000, "random_state": SEED, **p}),
        "knn": lambda p: KNeighborsClassifier(**{"n_neighbors": 25, **p}),
        "tree": lambda p: DecisionTreeClassifier(**{"random_state": SEED, **p}),
        "svm": lambda p: SVC(**{"kernel": "rbf", "C": 1.0, "probability": True, "random_state": SEED, **p}),
        "nb": lambda p: GaussianNB(**p),
        "rf": lambda p: RandomForestClassifier(**{"n_estimators": 300, "random_state": SEED, "n_jobs": 2, **p}),
        "hgb": lambda p: HistGradientBoostingClassifier(**{"random_state": SEED, **p}),
    }
    return reg[name](params)


# порядок «от простой к сложной» — для правила выбора при почти равных метриках
COMPLEXITY = ["dummy", "persistence", "logreg_l2", "logreg_l1", "nb", "tree", "knn", "svm", "rf", "hgb"]
