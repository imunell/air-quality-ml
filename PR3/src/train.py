"""Обучение и сохранение итоговой модели (практическая работа № 3).

Запуск одной командой:  python -m src.models.train
Читает выбранную конфигурацию из models/selected_model.json (её записывает ноутбук после
сравнения моделей), обучает Pipeline на всей обучающей выборке и сохраняет
models/model_v1.joblib и описание models/model_v1.json."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.models.experiments import SEED, build_model, split_train_test
from src.pipeline.features import file_sha256, make_dataset, make_pipeline


def main(initial="data/interim/initial_part.csv", out_dir="models", version="v1"):
    out = Path(out_dir)
    spec = json.loads((out / "selected_model.json").read_text(encoding="utf-8"))
    X, y, _ = make_dataset(initial)
    X_train, y_train, _, _ = split_train_test(X, y)          # тест не используется
    model = make_pipeline(build_model(spec["name"], **spec["params"]), spec["feature_set"])
    model.fit(X_train, y_train)
    joblib.dump(model, out / f"model_{version}.joblib")
    info = {
        "version": version, "trained": spec.get("trained"),
        "data": f"initial_part.csv (2007–2017), sha256 {file_sha256(initial)[:16]}…; "
                f"обучение: прогнозы на 2008–2015 годы, {len(X_train)} строк",
        "task": "прогноз уровня загрязнения (1–4) на год t+1 по данным года t; зона риска — уровни 3–4",
        "algorithm": f"{spec['name']} {spec['params']}", "feature_set": spec["feature_set"], "seed": SEED,
        "cv_scheme": "расширяющееся окно по годам, проверка на 2011–2015",
        **{k: spec[k] for k in spec if k.startswith("cv_")},
        "selection_rule": spec.get("selection_rule"),
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__},
    }
    (out / f"model_{version}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return model, info


if __name__ == "__main__":
    m, i = main(*sys.argv[1:])
    print("Модель сохранена:", i["algorithm"], "| CV ROC-AUC:", i.get("cv_roc_auc"))
