# Воспроизводимый запуск (практическая работа № 3)
PY ?= python

data:        ## разбиение данных на исходную часть и порции (ПР № 2)
	$(PY) src/data/split_batches.py

train:       ## обучение и сохранение итоговой модели
	$(PY) -m src.models.train

test:        ## автотесты пайплайна
	$(PY) -m pytest tests -q

repro: data train test
