# Dataset guide

This directory contains the generated train and reference-test datasets for the
surrogate-model benchmark. Every table is gzip-compressed, tab-separated, and
has the form:

```text
input_1    input_2    ...    target
```

The `target` column is noise-free simulator output. Symbolic-regression methods
receive only the input columns and target values; the ground-truth equations
below are not supplied during fitting. The goal is to discover a compact
expression that predicts the target on the independent test set and, ideally,
is algebraically equivalent or physically close to the generating equation.

Generated datasets are intentionally excluded from Git because they can be
reproduced from the scripts in `surrogate/`.

## Common dataset layout

| File | Samples | Purpose |
|---|---:|---|
| `data/<problem>/train.tsv.gz` | 400 | Fit the symbolic-regression model |
| `data/<problem>/test.tsv.gz` | 4,000 | Independent reference evaluation |

The train/test generation seeds are fixed. Cantilever uses independent uniform
random samples. Borehole and Piston use independently seeded scrambled Latin
hypercube samples for stratified coverage of every input dimension.

## 1. Cantilever beam tip deflection

Directory: `data/cantilever/`

The target is a dimensionless form of cantilever-beam tip deflection. The exact
expression being searched for is

$$
y = \frac{4 F L^3}{E b h^3},
$$

where the dataset columns map to the symbols as follows:

| Column | Symbol | Meaning | Range |
|---|---:|---|---:|
| `force` | $F$ | Applied force factor | $[0.5, 1.5]$ |
| `length` | $L$ | Beam length factor | $[0.8, 1.2]$ |
| `modulus` | $E$ | Elastic-modulus factor | $[0.8, 1.2]$ |
| `width` | $b$ | Beam width factor | $[0.7, 1.3]$ |
| `height` | $h$ | Beam height factor | $[0.7, 1.3]$ |

This is the simplest problem in the suite. It primarily tests multiplication,
division, feature relevance, and recovery of integer powers.

Generator: `surrogate/generate_cantilever.py`

## 2. Borehole water flow

Directory: `data/borehole/`

The target $y$ is water flow through a borehole in $\mathrm{m^3/year}$. Define

$$
\ell = \log\left(\frac{r}{r_w}\right).
$$

The exact expression being searched for is

$$
y =
\frac{2\pi T_u(H_u-H_l)}
{\ell\left(1 + \frac{2LT_u}{\ell r_w^2K_w} + \frac{T_u}{T_l}\right)}.
$$

| Column | Symbol | Meaning | Range |
|---|---:|---|---:|
| `borehole_radius` | $r_w$ | Borehole radius (m) | $[0.05, 0.15]$ |
| `influence_radius` | $r$ | Radius of influence (m) | $[100, 50000]$ |
| `upper_transmissivity` | $T_u$ | Upper-aquifer transmissivity ($\mathrm{m^2/year}$) | $[63070, 115600]$ |
| `upper_head` | $H_u$ | Upper-aquifer potentiometric head (m) | $[990, 1110]$ |
| `lower_transmissivity` | $T_l$ | Lower-aquifer transmissivity ($\mathrm{m^2/year}$) | $[63.1, 116]$ |
| `lower_head` | $H_l$ | Lower-aquifer potentiometric head (m) | $[700, 820]$ |
| `borehole_length` | $L$ | Borehole length (m) | $[1120, 1680]$ |
| `hydraulic_conductivity` | $K_w$ | Borehole hydraulic conductivity (m/year) | $[9855, 12045]$ |

This problem tests an eight-dimensional interaction containing subtraction,
ratios, a squared variable, and a logarithm across very different input scales.

Generator: `surrogate/generate_borehole.py`

## 3. Piston cycle time

Directory: `data/piston/`

The target $C$ is the piston cycle time in seconds. The ground truth is easiest
to state using two intermediate expressions:

$$
A = P_0S + 19.62M - \frac{kV_0}{S},
$$

$$
V = \frac{S}{2k}
\left(\sqrt{A^2 + 4k\frac{P_0V_0}{T_0}T_a} - A\right),
$$

followed by the exact target equation

$$
C = 2\pi\sqrt{
\frac{M}{k + S^2\frac{P_0V_0}{T_0}\frac{T_a}{V^2}}
}.
$$

| Column | Symbol | Meaning | Range |
|---|---:|---|---:|
| `piston_mass` | $M$ | Piston mass (kg) | $[30, 60]$ |
| `surface_area` | $S$ | Piston surface area ($\mathrm{m^2}$) | $[0.005, 0.020]$ |
| `initial_volume` | $V_0$ | Initial gas volume ($\mathrm{m^3}$) | $[0.002, 0.010]$ |
| `spring_coefficient` | $k$ | Spring coefficient (N/m) | $[1000, 5000]$ |
| `atmospheric_pressure` | $P_0$ | Atmospheric pressure ($\mathrm{N/m^2}$) | $[90000, 110000]$ |
| `ambient_temperature` | $T_a$ | Ambient temperature (K) | $[290, 296]$ |
| `gas_temperature` | $T_0$ | Filling-gas temperature (K) | $[340, 360]$ |

This is the most deeply nested target in the current suite. It tests whether an
algorithm can approximate or recover coupled division, subtraction, squares,
and square roots.

Generator: `surrogate/generate_piston.py`

## How results are judged

The current evaluator reports test-set RMSE, range-normalized RMSE, MAE, maximum
absolute error, $R^2$, runtime, and model size when the adapter exposes it. A
high $R^2$ alone does not prove equation recovery: two expressions may predict
similarly over these ranges while behaving differently elsewhere. Conversely,
an expression written in a different but algebraically equivalent form should
count as successful symbolic recovery. Predictive and symbolic-fidelity results
should therefore be reported separately in the thesis.
