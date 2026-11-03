PYTHON ?= python3
export OMP_NUM_THREADS = 1
export OPENBLAS_NUM_THREADS = 1
export MKL_NUM_THREADS = 1
export MPLCONFIGDIR ?= /tmp/concomitant-huber-mpl
.PHONY: sources optional-amp-check report check smoke pilot r4 r4-scaling rome-reference rome-validation rome-full diagnostics amp convergence tables manuscript pdf
sources:
	$(PYTHON) analysis/fetch_sources.py
optional-amp-check:
	$(PYTHON) analysis/check_optional_amp.py
report:
	$(PYTHON) analysis/record_primary_report.py
check:
	$(PYTHON) -m unittest discover -s tests -v
smoke:
	$(PYTHON) analysis/pipeline.py smoke
pilot:
	$(PYTHON) analysis/small_scale_experiment.py
r4:
	$(PYTHON) analysis/r4_benchmark.py
r4-scaling:
	$(PYTHON) analysis/r4_path_scaling.py
rome-reference:
	$(PYTHON) analysis/install_rome_reference.py
rome-validation:
	$(PYTHON) analysis/rome_paper_experiment.py
	$(PYTHON) analysis/assemble_path_report.py
rome-full:
	$(PYTHON) analysis/rome_paper_experiment.py --full
diagnostics: optional-amp-check
	$(PYTHON) analysis/convergence_amp_experiment.py
amp: optional-amp-check
	$(PYTHON) analysis/amp_source_protocol.py
convergence: optional-amp-check
	$(PYTHON) analysis/rome_convergence.py
tables:
	$(PYTHON) analysis/assemble_tables.py
manuscript:
	$(PYTHON) analysis/render_optimized.py html
pdf:
	$(PYTHON) analysis/render_optimized.py pdf

.PHONY: audit-paths audit-report
audit-paths:
	$(PYTHON) analysis/audit_incomplete_paths.py

audit-report:
	quarto render analysis/incomplete-path-audit.qmd --profile audit --to html
	$(PYTHON) analysis/audit_incomplete_paths.py --record-report

.PHONY: revision-timings revision-r4 revision-real revision-figures algorithm-template
algorithm-template:
	$(PYTHON) analysis/fetch_algorithm_template.py
revision-timings:
	$(PYTHON) analysis/revision_rome_parallel.py
revision-r4:
	$(PYTHON) analysis/revision_r4.py
revision-real:
	$(PYTHON) analysis/revision_real_parallel.py
revision-figures:
	$(PYTHON) analysis/revision_assemble.py

.PHONY: revision-amp
revision-amp: optional-amp-check
	$(PYTHON) analysis/revision_amp.py

.PHONY: optimized-cv optimized-pilot optimized-validation optimized-figures
optimized-cv:
	$(PYTHON) analysis/optimized_real_parallel.py
optimized-pilot:
	$(PYTHON) analysis/optimization_pilot.py
optimized-validation:
	$(PYTHON) analysis/optimized_timing_validation.py
optimized-figures:
	$(PYTHON) analysis/optimized_assemble.py

.PHONY: python-scale-timing
python-scale-timing:
	$(PYTHON) analysis/python_scale_timing.py

.PHONY: scale-strategies
scale-strategies:
	$(PYTHON) analysis/scale_strategy_comparison.py

.PHONY: scale-path-benchmark
scale-path-benchmark:
	$(PYTHON) analysis/scale_path_benchmark.py
	$(PYTHON) analysis/scale_path_report.py

.PHONY: figure1-complete
figure1-complete:
	$(PYTHON) analysis/figure1_complete.py
	$(PYTHON) analysis/figure1_accuracy.py
	$(PYTHON) analysis/optimized_assemble.py

.PHONY: implementation-audit
implementation-audit:
	$(PYTHON) analysis/implementation_audit.py
	quarto render analysis/implementation-audit.qmd --to html
	$(PYTHON) analysis/implementation_audit.py --record-html
