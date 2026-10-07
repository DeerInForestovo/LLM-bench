# LLM-bench experiment pipeline.
#
#   make data        -> download raw logos into data/raw/
#   make preprocess  -> clean + normalize logos into data/processed/
#   make benchmark   -> evaluate LCD scores and render the leaderboard
#   make all         -> full pipeline from scratch

PYTHON ?= python3

.PHONY: all data preprocess benchmark clean

all: data preprocess benchmark

data:
	bash scripts/download_data.sh

preprocess:
	$(PYTHON) src/preprocess.py

benchmark:
	$(PYTHON) benchmark.py

clean:
	rm -rf data/processed results
