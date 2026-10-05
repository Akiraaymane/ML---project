.PHONY: install train predict test clean

install:
	pip install -e ".[dev]"

train:
	python -m churnprofit.train --config configs/config.yaml

predict:
	python -m churnprofit.predict --input $(INPUT) --output targeting.csv

test:
	pytest -q

clean:
	rm -rf artifacts reports .pytest_cache
