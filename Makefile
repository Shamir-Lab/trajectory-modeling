PYTHON ?= python
PYTHONPATH := src
export PYTHONPATH

.PHONY: help install-bayes install-bootstrap install-evaluation install-all \
	traj-hirid-aki traj-hirid-sepsis traj-hirid-liver traj-hirid-ventilator \
	analysis-mimic-sepsis analysis-hirid-liver analysis-eicu-aki \
	launchers-generate subset-smoke slurm-check-hardcoded slurm-check-dataroot \
	clean veryclean

help:
	@echo "Trajectory Modeling — common commands"
	@echo ""
	@echo "  make install-bayes       - install package with Bayesian extras"
	@echo "  make install-bootstrap   - install package with Bootstrap extras"
	@echo "  make install-evaluation  - install package with evaluation extras"
	@echo "  make install-all         - install all extras"
	@echo ""
	@echo "  make traj-hirid-aki      - run HiRiD AKI trajectory script"
	@echo "  make traj-hirid-sepsis   - run HiRiD sepsis trajectory script"
	@echo "  make traj-hirid-liver    - run HiRiD liver trajectory script"
	@echo "  make traj-hirid-ventilator - run HiRiD ventilator trajectory script"
	@echo ""
	@echo "  make analysis-mimic-sepsis - run MIMIC sepsis analysis (quick test)"
	@echo "  make analysis-hirid-liver  - run HiRiD liver analysis (quick test)"
	@echo "  make analysis-eicu-aki     - run eICU AKI analysis (quick test)"
	@echo "  make launchers-generate    - regenerate launcher scripts"
	@echo "  make subset-smoke        - dry-run subset Bayesian runner"
	@echo "  make slurm-check-hardcoded - detect hardcoded repo paths in SLURM scripts"
	@echo "  make slurm-check-dataroot  - list DATA_ROOT defaults in SLURM scripts"
	@echo ""
	@echo "  make clean               - remove Python cache files"
	@echo "  make veryclean           - clean + remove local logs/results artifacts"

install-bayes:
	$(PYTHON) -m pip install -e .[bayes]

install-bootstrap:
	$(PYTHON) -m pip install -e .[bootstrap]

install-evaluation:
	$(PYTHON) -m pip install -e .[evaluation]

install-all:
	$(PYTHON) -m pip install -e .[all]

traj-hirid-aki:
	$(PYTHON) scripts/trajectory/hirid/aki_trajs.py

traj-hirid-sepsis:
	$(PYTHON) scripts/trajectory/hirid/sepsis_trajs.py

traj-hirid-liver:
	$(PYTHON) scripts/trajectory/hirid/liver_trajs.py

traj-hirid-ventilator:
	$(PYTHON) scripts/trajectory/hirid/ventilator_trajs.py

analysis-mimic-sepsis:
	$(PYTHON) scripts/analysis/mimic/mimic_sepsis_analysis.py --cv-repeats 1 --cv-splits 2

analysis-hirid-liver:
	$(PYTHON) scripts/analysis/hirid/hirid_liver_analysis.py --cv-repeats 1 --cv-splits 2

analysis-eicu-aki:
	$(PYTHON) scripts/analysis/eicu/eicu_aki_analysis.py --cv-repeats 1 --cv-splits 2

launchers-generate:
	bash scripts/launchers/generate_launchers.sh

subset-smoke:
	$(PYTHON) scripts/trajectory/run_circulatory_failure_bayes_subset.py --dry-run

slurm-check-hardcoded:
	@echo "Checking for hardcoded repo path in scripts/slurm..."
	@grep -R --line-number "/home/gaga/tamarw1/trajectory-modeling" scripts/slurm || echo "OK: no hardcoded repo path found"

slurm-check-dataroot:
	@echo "DATA_ROOT defaults currently present in scripts/slurm:"
	@grep -R --line-number "DATA_ROOT=" scripts/slurm || true

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

veryclean: clean
	rm -rf logs/outs logs/errs
	rm -rf results/eicu results/hirid results/mimic results/smoke_shared_template
