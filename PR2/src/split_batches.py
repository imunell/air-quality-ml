"""Разбиение набора на исходную часть и порции «новых» данных.

Набор содержит год наблюдения (year), поэтому делим по времени:
  - исходная часть (initial) — первые годы, около 60 % записей;
  - порции (batch_XX.csv) — последующие годы, имитация ежегодных выгрузок.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/data_air_cities_100_v20260617.csv")
INTERIM = Path("data/interim/initial_part.csv")
INCOMING = Path("data/incoming")
MANIFEST = Path("data/MANIFEST.json")
SOURCE = "Росгидромет / ГГО им. Воейкова, https://tochno.st/datasets/air_cities (CC BY)"

INITIAL_LAST_YEAR = 2017 #2007–2017 ~61 % записей (ближайшая к 60 % годовая граница)
BATCHES = [(2018, 2019), (2020, 2021), (2022, 2023), (2024, 2024)]  #4 порции: выгрузки за 1–2 года


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    df = pd.read_csv(RAW, encoding="utf-8-sig", dtype=str)
    year = df["year"].astype(int)
    INTERIM.parent.mkdir(parents=True, exist_ok=True)
    INCOMING.mkdir(parents=True, exist_ok=True)

    parts = {INTERIM: df[year <= INITIAL_LAST_YEAR]}
    for i, (a, b) in enumerate(BATCHES, 1):
        parts[INCOMING / f"batch_{i:02d}.csv"] = df[(year >= a) & (year <= b)]

    manifest = [{"file": str(RAW), "rows": len(df), "sha256": sha256(RAW), "source": SOURCE}]
    for path, part in parts.items():
        part.to_csv(path, index=False, encoding="utf-8")
        manifest.append({"file": str(path), "rows": len(part), "sha256": sha256(path),
                         "years": f"{part['year'].min()}–{part['year'].max()}", "source": str(RAW)})
    assert sum(len(p) for p in parts.values()) == len(df), "потеряны или задвоены записи"
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for m in manifest:
        print(m["file"], m["rows"], m["sha256"][:12])


if __name__ == "__main__":
    main()
