# SurrogateModel_SRBench

Bachelor-thesis benchmarking project for evaluating symbolic-regression
algorithms as surrogate models.

This repository starts from the experiment interface of
[SRBench](https://github.com/cavalab/srbench). The initial import is deliberately
small: it contains the evaluator, two bundled test datasets, the `gplearn`
symbolic-regression adapter, and lightweight scikit-learn baselines. Large PMLB
datasets, result archives, notebooks, and heavyweight algorithm environments are
not included.

## Quick start

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-dev.txt
./scripts/smoke_test.sh
```

The smoke test verifies the common estimator interface and performs a short fit,
prediction, metric, and symbolic-expression round trip with `gplearn`. It also
tests a linear-regression baseline through the same evaluator.

## Surrogate problems

The first end-to-end problem approximates the dimensionless tip deflection of a
cantilever beam:

`deflection = 4 * force * length^3 / (modulus * width * height^3)`

Training and test samples are generated independently with a fixed seed. Run the
official SRBench gplearn image with:

```bash
./scripts/run_cantilever_docker.sh
```

The script pulls the immutable official image recorded in
`containers/images.lock`, records the resolved digest with the result, mounts
this repository, and writes results under `results/cantilever/`. Set
`SRBENCH_IMAGE` to deliberately override the locked image. A local control run
is available as `./scripts/run_cantilever_local.sh`.

Run the initial diverse algorithm group (`gplearn`, Operon, PySR,
GeneticEngine, ITEA, and EQL) with reduced smoke-test budgets using:

```bash
./scripts/run_algorithm_group.sh
```

An individual pinned image can be run with
`./scripts/run_algorithm_docker.sh ALGORITHM`. These smoke settings establish
compatibility; they are not the final thesis benchmark budgets.

Run the versioned three-seed pilot benchmark with:

```bash
./scripts/run_benchmark_group.sh
```

The original Cantilever-only pilot remains defined in
`configs/benchmark_pilot_v1.json`; the three-problem suite is defined in
`configs/benchmark_suite_v1.json`. The current evaluation protocol, including
repeated prediction timing, expression analysis, and explicitly separated
raw/normalized input conditions, is versioned as
`configs/benchmark_suite_v3.json` and is the runner default. Individual trials
use `./scripts/run_benchmark_trial.sh ALGORITHM SEED [PROBLEM] [INPUT_SCALING]`,
and aggregate metrics are written to
`results/<problem>/<benchmark-name>/<input-scaling>/summary.csv`. The summary
also reports expected, successful, and missing trials so incomplete experiment
matrices are visible.

The full suite adds the established Borehole water-flow and Piston cycle-time
computer-experiment problems. Their selection rationale, equations, domains,
and sources are documented in [docs/surrogate-problems.md](docs/surrogate-problems.md).
The generated table layout, every input column, and the exact target equations
are summarized in [data/README.md](data/README.md).
Framework-independent expression complexity, symbolic ground-truth comparison,
and runtime measurement are defined in
[docs/evaluation-protocol.md](docs/evaluation-protocol.md).
By default, `run_benchmark_group.sh` runs all three problems in both `raw` and
`domain_minmax` conditions, with ten repetition labels (42–51): 360 expected
trials in total. Restrict a run with, for example,
`BENCHMARK_PROBLEMS="borehole piston"` or `BENCHMARK_SCALINGS=domain_minmax`.
Run one normalized trial with:

```bash
./scripts/run_benchmark_trial.sh gplearn 42 borehole domain_minmax
```

Normalization maps each published physical domain to $[-1,1]$ using fixed
bounds; it is never fitted from the sampled data, and the target is not scaled.
Every result JSON records the condition, formula, and bounds. When expression
parsing succeeds, a model learned in normalized coordinates is also transformed
back to the original physical variables for interpretation and comparison with
the generating equation.

The generated train/test tables are fixed across repetitions and their content
hashes are stored in every result. Supported estimators receive the repetition
seed; ITEA is explicitly summarized as having uncontrolled repetitions because
its published estimator exposes no seed parameter. Failed runs receive separate
failure records, and summaries distinguish successful, failed, and missing
trials.

The published Operon and GeneticEngine images contain older package APIs than
the current SRBench adapters. Their adapters include narrowly scoped fallback
paths for those pinned images. ITEA's published estimator exposes no random-seed
parameter; result JSON records this explicitly as `"seed_parameter": null`.

On a supported Ubuntu system without Docker, install Docker Engine from its
official apt repository with:

```bash
./scripts/install_docker_ubuntu.sh
```

The command requires interactive administrator authentication. Log out and back
in afterward so membership in the `docker` group takes effect.

To test an individual adapter from the `experiment` directory:

```bash
cd experiment
python3 -m pytest -v test_algorithm.py --ml gplearn
python3 -m pytest -v test_evaluate_model.py --ml sklearn_linear
```

## Included upstream components

- `experiment/`: SRBench evaluation and test code
- `experiment/methods/gplearn/`: symbolic-regression adapter
- `experiment/methods/sklearn_*`: inexpensive baseline adapters
- `experiment/test/`: small smoke-test datasets
- `algorithms/`: installation metadata for the included method families
- `docs/`: archived upstream README and user guide

The exact imported SRBench revision is recorded in
`SRBENCH_UPSTREAM_COMMIT`. The original repository is configured as the Git
remote named `upstream`.

## Next project step

The existing SRBench evaluator uses a random 75/25 split. Before running thesis
experiments, add a surrogate-specific evaluator that accepts fixed training and
reference test sets so that design-of-experiments samples cannot leak into the
test set.

## License and attribution

This project is derived from SRBench and is licensed under GPL-3.0. The original
license and contributor notices are retained. Individual integrated algorithms
may have their own licenses and must be checked before redistribution.
