"""Пайплайн подготовки данных и признаков (практическая работа № 3).

Задача: по данным города за год t спрогнозировать общий уровень загрязнения
воздуха (air_general_level, 1–4) в следующем году t+1.

Состав модуля
  clean(df)         очистка без параметров (шаги из таблицы 9 практической работы № 2)
  add_history(df)   лаговые признаки по истории города (нужна таблица с прошлыми годами)
  add_features(df)  построчные признаки (шаг sklearn Pipeline)
  make_dataset()    одной командой: файл -> (X, y, meta), готовые для модели
  make_pipeline()   предобработка + модель в одном объекте Pipeline
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- константы
EM = ["air_solid_emissions", "air_so_emissions", "air_no_emissions", "air_co_emissions"]
KEY = ["region", "city"]
TARGET = "target_next"          # уровень загрязнения в году t+1
LEVELS = [1, 2, 3, 4]
HIGH_FROM = 3                   # «зона риска» — уровни 3 и 4

# Допустимые диапазоны (требования Q-05…Q-07 практической работы № 2)
EM_RANGE = (0.0, 5000.0)        # тыс. т
POP_RANGE = (0.0, 15000.0)      # тыс. чел. (строго больше 0)
LAT_RANGE, LON_RANGE = (41.0, 82.0), (19.0, 180.0)

# Наборы признаков для оценки пользы новых признаков (раздел 3.8)
# A — исходные столбцы года t; B — A + история города; C — B + остальные новые признаки
FEATURES_A_NUM = EM + ["air_population", "n_st_main", "geo_lat", "geo_lon"]
FEATURES_B_NUM = FEATURES_A_NUM + ["level_lag1", "high_share_3y", "n_qcp", "n_si"]
FEATURES_C_NUM = FEATURES_B_NUM + ["emis_total", "emis_per_cap", "emis_growth"]
FEATURES_C_CAT = ["pop_group", "norm_regime"]
# Признаки, к которым применяется log1p (сильно скошенные, неотрицательные)
LOG_COLS = EM + ["air_population", "emis_total", "emis_per_cap"]

FEATURE_SETS = {
    "A": (FEATURES_A_NUM, []),
    "B": (FEATURES_B_NUM, []),
    "C": (FEATURES_C_NUM, FEATURES_C_CAT),
    "D": (FEATURES_C_NUM, FEATURES_C_CAT + ["region"]),   # C + регион (проверка)
}


# ---------------------------------------------------------------- загрузка
def load_raw(path) -> pd.DataFrame:
    """Чтение файла (UTF-8, коды как строки) — шаг «Загрузка»."""
    return pd.read_csv(path, encoding="utf-8-sig",
                       dtype={"region_oktmo": str, "city_oktmo": str, "fias_code": str})


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- очистка
def _count_substances(s: pd.Series) -> pd.Series:
    """Число веществ в текстовом поле («БП, NO2, Ф» -> 3; «No substance» -> 0)."""
    s = s.fillna("No substance").astype(str)
    n = s.str.split(",").apply(lambda parts: sum(1 for p in parts if p.strip()))
    return n.where(s.str.strip() != "No substance", 0)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка без параметров, оцениваемых по данным (поэтому одинакова для обучения и порции)."""
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["city_key"] = df["region"].astype(str) + " | " + df["city"].astype(str)   # city_oktmo неуникален
    df = df.drop_duplicates(subset=["city_key", "year"], keep="last")
    # цель: -9 («не оценён») и значения вне {1..4} -> пропуск
    lvl = pd.to_numeric(df["air_general_level"], errors="coerce")
    df["air_general_level"] = lvl.where(lvl.isin(LEVELS))
    # значения вне допустимых диапазонов -> пропуск
    for c in EM:
        v = pd.to_numeric(df[c], errors="coerce")
        df[c] = v.where((v >= EM_RANGE[0]) & (v <= EM_RANGE[1]))
    v = pd.to_numeric(df["air_population"], errors="coerce")
    df["air_population"] = v.where((v > POP_RANGE[0]) & (v <= POP_RANGE[1]))
    la, lo = pd.to_numeric(df["geo_lat"], errors="coerce"), pd.to_numeric(df["geo_lon"], errors="coerce")
    ok = la.between(*LAT_RANGE) & lo.between(*LON_RANGE)
    df["geo_lat"], df["geo_lon"] = la.where(ok), lo.where(ok)
    # число станций основной сети: «5+1» -> 5, «эп» -> NaN
    df["n_st_main"] = pd.to_numeric(df["air_stantions"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
    # счётчики веществ из текстовых полей
    df["n_qcp"] = _count_substances(df["air_qcp"])
    df["n_si"] = _count_substances(df["air_standard_index"])
    return df.sort_values(["city_key", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------- история города
def _shifted(df: pd.DataFrame, col: str, k: int, name: str) -> pd.Series:
    """Значение столбца в году (t − k) для того же города (по ключу, а не по номеру строки:
    у города могут быть пропущенные годы)."""
    src = df[["city_key", "year", col]].copy()
    src["year"] = src["year"] + k
    m = df[["city_key", "year"]].merge(src.rename(columns={col: name}), on=["city_key", "year"], how="left")
    return m[name].to_numpy()


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Лаговые признаки. Использует только годы ≤ t, поэтому не раскрывает будущее.
    Для порции в df передаётся история из хранилища вместе с новыми строками."""
    df = df.copy()
    df["emis_total"] = df[EM].sum(axis=1, min_count=1)
    high = (df["air_general_level"] >= HIGH_FROM).astype(float).where(df["air_general_level"].notna())
    df["_high"] = high
    df["level_lag1"] = df["air_general_level"]                 # уровень года t (признак для прогноза t+1)
    lag = [df["_high"].to_numpy(), _shifted(df, "_high", 1, "h1"), _shifted(df, "_high", 2, "h2")]
    df["high_share_3y"] = pd.DataFrame(lag).T.mean(axis=1, skipna=True).to_numpy()   # доля лет 3–4 за t, t-1, t-2
    df["emis_total_prev"] = _shifted(df, "emis_total", 1, "ep")
    # цель: уровень в году t+1 (только для обучения; для новой порции остаётся пропуском)
    df[TARGET] = _shifted(df, "air_general_level", -1, "tn")
    return df.drop(columns=["_high"])


# ---------------------------------------------------------------- построчные признаки
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Построчные признаки; функция без параметров -> FunctionTransformer."""
    df = df.copy()
    if "emis_total" not in df:
        df["emis_total"] = df[EM].sum(axis=1, min_count=1)
    pop = df["air_population"]
    df["emis_per_cap"] = df["emis_total"] / pop                         # т на жителя (тыс. т / тыс. чел.)
    prev = df["emis_total_prev"].replace(0, np.nan) if "emis_total_prev" in df else np.nan
    df["emis_growth"] = (df["emis_total"] / prev - 1).clip(-1, 5)       # относительный прирост выбросов
    df["pop_group"] = pd.cut(pop, [0, 50, 100, 250, 500, np.inf], right=False,
                             labels=["<50", "50-100", "100-250", "250-500", ">=500"]).astype(object)
    ty = df["year"] + 1                                                # год, на который строится прогноз
    df["norm_regime"] = np.select([ty <= 2013, ty <= 2020], ["до 2014", "2014-2020"], default="с 2021")
    return df


# ---------------------------------------------------------------- набор данных одной командой
def make_dataset(path_or_df, history: pd.DataFrame | None = None):
    """Файл (или таблица) -> (X, y, meta), готовые для модели.

    history — очищенная таблица прошлых лет (для новой порции). Результат детерминирован:
    два запуска на одних данных дают идентичные таблицы (проверяется хэшем в tests/)."""
    raw = load_raw(path_or_df) if isinstance(path_or_df, (str, Path)) else path_or_df
    cur = clean(raw)
    full = cur if history is None else clean(pd.concat([history_raw(history), raw], ignore_index=True))
    full = add_history(full)
    if history is not None:                      # оставляем только строки новой порции
        keys = set(zip(cur["city_key"], cur["year"]))
        full = full[[k in keys for k in zip(full["city_key"], full["year"])]]
    full = full.reset_index(drop=True)
    meta = full[["city_key", "region", "city", "year"]].copy()
    X = full.drop(columns=[TARGET])
    y = full[TARGET]
    return X, y, meta


def history_raw(h: pd.DataFrame) -> pd.DataFrame:
    return h[[c for c in h.columns if c in _RAW_COLS]]


_RAW_COLS = ["year", "region", "region_oktmo", "city", "city_oktmo", "name_with_type", "geo_lat", "geo_lon",
             "fias_id", "fias_code", "air_general_level", "air_standard_index", "air_repeatability", "air_qcp",
             "air_solid_emissions", "air_so_emissions", "air_no_emissions", "air_co_emissions",
             "air_population", "air_stantions"]


def table_hash(df: pd.DataFrame) -> str:
    """SHA-256 таблицы (для проверки повторяемости)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
                          + "|".join(map(str, df.columns)).encode()).hexdigest()


# ---------------------------------------------------------------- Pipeline
def make_pipeline(model, feature_set: str = "C") -> Pipeline:
    """Расчёт признаков + предобработка + модель в одном объекте.
    Параметры предобработки (медианы, масштаб, категории) оцениваются при fit
    заново на каждом обучающем блоке кросс-валидации — утечки через предобработку нет."""
    num, cat = FEATURE_SETS[feature_set]
    log_cols = [c for c in num if c in LOG_COLS]
    lin_cols = [c for c in num if c not in LOG_COLS]
    log_branch = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                           ("log", FunctionTransformer(_log1p_keep, feature_names_out="one-to-one")),
                           ("scale", StandardScaler())])
    lin_branch = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                           ("scale", StandardScaler())])
    cat_branch = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    parts = [("num_log", log_branch, log_cols), ("num", lin_branch, lin_cols)]
    if cat:
        parts.append(("cat", cat_branch, cat))
    prep = ColumnTransformer(parts, sparse_threshold=0.0, verbose_feature_names_out=True)
    return Pipeline([("features", FunctionTransformer(add_features)),
                     ("prep", prep), ("model", model)])


def _log1p_keep(a):
    """log1p для неотрицательных значений; индикаторы пропусков (0/1) log1p не портит монотонностью."""
    return np.log1p(np.clip(a, 0, None))
