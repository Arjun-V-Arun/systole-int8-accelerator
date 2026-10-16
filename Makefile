# Top-level entry points. Each target is independent.

.PHONY: all model lint test regress mutate formal synth perf clean

all: lint model test

model:            ## build the C++ model, its self-test and the perf study
	$(MAKE) -C model all
	./model/selftest

lint:             ## Verilator lint, several configurations
	@for P in "" "-GPIPE_MAC=1" "-GROWS=8 -GCOLS=8" "-GROWS=3 -GCOLS=5 -GPIPE_MAC=1" "-GROWS=1 -GCOLS=1"; do \
	  verilator --lint-only -Wall -DSA_CHECKS_ON $$P rtl/pe.sv rtl/systolic_array.sv rtl/sa_top.sv --top-module sa_top || exit 1; \
	  echo "lint clean: $$P"; \
	done

test:             ## quick regression: 4x4 and pipelined 4x4
	python3 tb/run_regression.py --sizes 4x4,4x4p --jobs 100

regress:          ## full regression across array shapes
	python3 tb/run_regression.py --sizes 4x4,8x8,3x5,1x1,4x4p,8x8p,3x5p,1x1p,16x16 --jobs 300

mutate:           ## inject bugs and check the testbench catches them
	python3 scripts/mutation_test.py

formal:           ## k-induction proof of the RTL assertions
	./formal/run_formal.sh

synth:            ## SKY130 synthesis (fetches the liberty file if missing)
	@test -f synth/lib/sky130_fd_sc_hd__tt_025C_1v80.lib || ./synth/get_lib.sh
	./synth/run_synth.sh 4 4 0 0
	./synth/run_synth.sh 4 4 1 0
	./synth/run_synth.sh 4 4 0 1000
	./synth/run_synth.sh 4 4 1 1000
	./synth/run_synth.sh 8 8 0 1000
	./synth/run_synth.sh 8 8 1 1000
	python3 scripts/synth_summary.py

perf:             ## dataflow study on a GPT-2 small transformer block
	$(MAKE) -C model perf_model
	./model/perf_model docs/perf_results.csv > docs/perf_results.md
	cat docs/perf_results.md

clean:
	rm -rf sim_build results_*.xml tb/__pycache__ synth/out formal/logs
	$(MAKE) -C model clean
