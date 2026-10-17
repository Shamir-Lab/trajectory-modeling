PYTHON ?= python
PYTHONPATH := src
export PYTHONPATH

.PHONY: help install-bayes install-bootstrap install-evaluation install-all \
	test test-all analysis-mimic-sepsis analysis-eicu-aki \
	slurm-check-hardcoded slurm-check-dataroot clean

help:
	@echo "Trajectory Modeling — common commands"
	@echo ""
	@echo "  make install-bayes       - install package with Bayesian extras"
	@echo "  make install-bootstrap   - install package with Bootstrap extras"
	@echo "  make install-evaluation  - install package with evaluation extras"
	@echo "  make install-all         - install all extras"
	@echo ""
	@echo "  make test                - run the fast smoke tests"
	@echo "  make test-all            - run all smoke tests, including MCMC sampling"
	@echo ""
	@echo "  make analysis-mimic-sepsis - run MIMIC sepsis analysis (quick test)"
	@echo "  make analysis-eicu-aki     - run eICU AKI analysis (quick test)"
	@echo "  make slurm-check-hardcoded - detect hardcoded repo paths in SLURM scripts"
	@echo "  make slurm-check-dataroot  - list DATA_ROOT defaults in SLURM scripts"
	@echo ""
	@echo "  make clean               - remove Python cache files"

install-bayes:
	$(PYTHON) -m pip install -e .[bayes]

install-bootstrap:
	$(PYTHON) -m pip install -e .[bootstrap]

install-evaluation:
	$(PYTHON) -m pip install -e .[evaluation]

install-all:
	$(PYTHON) -m pip install -e .[all]

test:
	$(PYTHON) -m pytest -q tests/test_smoke_backends.py -m "not slow"

test-all:
	$(PYTHON) -m pytest -q tests/test_smoke_backends.py -m "not slow or slow"

analysis-mimic-sepsis:
	$(PYTHON) scripts/analysis/mimic/mimic_sepsis_analysis.py --cv-repeats 1 --cv-splits 2

analysis-eicu-aki:
	$(PYTHON) scripts/analysis/eicu/eicu_aki_analysis.py --cv-repeats 1 --cv-splits 2

slurm-check-hardcoded:
	@echo "Checking for hardcoded repo path in scripts/slurm..."
	@grep -R --line-number "/home/gaga/tamarw1/trajectory-modeling" scripts/slurm || echo "OK: no hardcoded repo path found"

slurm-check-dataroot:
	@echo "DATA_ROOT defaults currently present in scripts/slurm:"
	@grep -R --line-number "DATA_ROOT=" scripts/slurm || true

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
