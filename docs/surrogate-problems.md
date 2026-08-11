# Surrogate problem selection

The benchmark suite deliberately combines three deterministic engineering
problems with known analytical ground truth. This makes it possible to measure
both predictive accuracy and, later, symbolic recovery. All problems use fixed,
independent training and reference-test designs with 400 and 4,000 samples.

## Problems

### Cantilever tip deflection

The existing five-input problem is a comparatively accessible rational power
law. It tests whether a method can recover multiplicative structure and integer
powers. Its dimensionless input domain around a nominal design is documented in
`surrogate/generate_cantilever.py`.

### Borehole flow

The Borehole function is a standard eight-input computer-experiment benchmark
that models water flow through a borehole. It adds a logarithm, interacting
ratios, disparate physical scales, and a higher input dimension. We use the
published standard uniform bounds and the original high-fidelity equation.

- Function and ranges: <https://www.sfu.ca/~ssurjano/borehole.html>
- Reference implementation: <https://www.sfu.ca/~ssurjano/Code/boreholer.html>
- Original source: Harper and Gupta (1983), *Sensitivity/uncertainty analysis
  of a borehole scenario comparing Latin Hypercube Sampling and deterministic
  sensitivity approaches*.

### Piston cycle time

The seven-input Piston simulation models the cycle time of a piston in a
cylinder. Its nested intermediate volume calculation combines subtraction,
division, squared terms, and square roots. This creates a materially different
symbolic-recovery challenge from the rational Cantilever and Borehole problems.

- Function and ranges: <https://www.sfu.ca/~ssurjano/piston.html>
- Reference implementation: <https://www.sfu.ca/~ssurjano/Code/pistonr.html>
- Reference: Ben-Ari and Steinberg (2007), “Modeling data from computer
  experiments: an empirical comparison of kriging with MARS and projection
  pursuit regression,” *Quality Engineering*, 19(4), 327–338.

## Experimental design

Borehole and Piston use seeded, scrambled Latin hypercube designs. Each
univariate marginal is stratified, which gives more even coverage than an
equally sized independent random sample. Training and test designs use separate
seeds so no points are shared. SciPy documents the method and its underlying
references at
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html>.

The pre-existing Cantilever generator is intentionally unchanged so the
completed `benchmark_pilot_v1` remains reproducible. A future final thesis
protocol can migrate all three problems to one common design policy under a new
versioned configuration.

The SFU reference-code pages carry GPL-2.0 notices. No source code from those
implementations is copied here: the vectorized Python functions independently
implement the published mathematical definitions. The links and authorship are
retained for provenance and numerical verification.
