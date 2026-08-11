# Complexity, interpretability, and runtime evaluation

This document defines how the benchmark evaluates symbolic expressions and
execution time. Predictive accuracy remains necessary, but it is not used as a
substitute for interpretability or symbolic recovery.

## Expression representations

Each result retains the adapter's original `symbolic_model`. The common
evaluator then parses that string with SymPy. Its parser supports the infix
notation returned by PySR, Operon, GeneticEngine, ITEA, and EQL, as well as the
prefix `add`, `sub`, `mul`, and `div` notation returned by gplearn.

Parsing and simplification have separate success and error fields. A timeout or
failure therefore remains visible and does not cause the complete model run to
be discarded. Symbolic processing has a default five-second limit per stage.

## Structural complexity

The following framework-independent measures are computed from the normalized
SymPy tree:

- `expression_node_count`: all operator, variable, and constant nodes;
- `expression_depth`: longest root-to-leaf edge count, with atoms at depth zero;
- `expression_operator_count` and `expression_operators`;
- `expression_constant_count`;
- `expression_variable_count` and `expression_variables`.

The same measures are recorded with a `simplified_` prefix for the smallest
candidate among the normalized, cancelled, factored, and generally simplified
forms. The adapter-specific `model_size` is retained
for debugging and comparison with upstream reports, but it is not suitable as
the primary cross-framework measure because adapters use different definitions.

SymPy canonicalizes associative operations, so normalized node counts measure
the mathematical expression tree rather than the framework's internal genome
or serialization syntax. Complexity is evidence about readability, not an
automatic interpretability verdict: a small but inaccurate or physically
implausible equation is not considered a good surrogate.

## Ground-truth comparison

The evaluator contains the exact Cantilever, Borehole, and Piston equations. It
records their structural measures alongside every discovered expression and
reports:

- `symbolic_exact_match`: whether their algebraic difference reduces exactly
  to zero;
- `ground_truth_variable_recall`: fraction of ground-truth variables present;
- `simplified_to_ground_truth_size_ratio`: simplified discovered size divided
  by ground-truth size.

Exact matching is deliberately strict. An equation with rounded coefficients
can have excellent predictive accuracy while receiving `false`. Final analysis
must therefore report symbolic matching together with test NRMSE and the actual
expression. Size ratio is descriptive and must not reward an inaccurate model
merely for being shorter than the true equation.

## Runtime measurements

All timings use Python's monotonic high-resolution `perf_counter` inside the
already running algorithm container:

- `fit_seconds`: the estimator's complete `fit` call;
- `prediction_seconds`: first prediction over the reference test set;
- `prediction_mean_seconds`, `prediction_median_seconds`, and
  `prediction_std_seconds`: repeated complete-test predictions;
- `prediction_microseconds_per_sample`: median prediction time divided by the
  test-set size;
- `expression_extraction_seconds`: adapter expression export and native size;
- `expression_analysis_seconds`: parsing, simplification, and comparison;
- `evaluation_seconds`: fit through completed expression analysis.

The suite configuration requests five prediction calls. The first-call value
captures initialization latency; the median is the preferred steady-use
comparison because individual sub-millisecond measurements are noisy.

Container startup, image pulling, dataset generation, JSON serialization, and
one-time installation are excluded. All benchmark containers receive one-thread
environment settings, but training times still describe algorithm-specific
configured budgets rather than equal-compute performance. They must not be
presented as a hardware-normalized speed ranking.

Results are isolated under `results/<problem>/<benchmark-name>/`, so changing a
versioned evaluation protocol cannot silently overwrite results produced by an
older configuration.

## Recommended reporting

For every problem and algorithm, report predictive error, fit time, median
prediction time, simplified node count, depth, variable recall, parsing success,
and exact-match rate over repeated trials. The CSV aggregator records mean,
sample standard deviation, median, minimum, and maximum for numeric measures.
An accuracy-versus-complexity scatter
or Pareto plot should use test NRMSE and simplified node count. Failed parsing,
missing expressions, failed runs, and algorithms without controllable seeds
must remain explicit in tables and captions.
