PYTHON ?= python3

.PHONY: install test smoke controlled telecomts multiseed verify reproduce clean

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) -m benchmark.run_benchmark --quick --outdir outputs/smoke_controlled
	$(PYTHON) -m benchmark.run_telecomts_benchmark --samples 80 --skip-checksum --outdir outputs/smoke_telecomts

controlled:
	$(PYTHON) -m benchmark.run_benchmark --timesteps 1800 --window 12 --horizon 3 --seed 7

telecomts:
	$(PYTHON) -m benchmark.run_telecomts_benchmark --samples 800 --input-len 96 --seed 17

multiseed:
	$(PYTHON) -m benchmark.run_multiseed --seeds 7 11 17 23 29 --samples 800 --input-len 96

verify:
	$(PYTHON) -m benchmark.verify_results

reproduce:
	./scripts/reproduce_paper.sh

clean:
	rm -rf outputs
