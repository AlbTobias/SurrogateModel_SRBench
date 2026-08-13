# Dataset guide

This directory contains the generated train and reference-test datasets for the
surrogate-model benchmark. Every table is gzip-compressed, tab-separated, and
has the form:

```text
input_1    input_2    ...    target
```

For Cantilever, Borehole, and Piston, the `target` column is noise-free simulator
output. CCPP instead contains measured power-plant observations. Symbolic-
regression methods receive only the input columns and target values. The goal is
to discover a compact expression that predicts the independent test set. Exact
symbolic recovery is additionally evaluated only where a generating equation is
known.

Generated datasets are intentionally excluded from Git because they can be
reproduced from the scripts in `surrogate/`.

All stored tables contain raw inputs in the physical units and ranges documented
below. Scaling is performed only inside the evaluator. The normalized benchmark
condition uses the fixed documented bounds to map every feature to $[-1,1]$;
it does not rewrite these files or scale the target.

## Common dataset layout

| File | Samples | Purpose |
|---|---:|---|
| `data/<problem>/train.tsv.gz` | 400 | Fit the symbolic-regression model |
| `data/<problem>/test.tsv.gz` | 4,000 | Independent reference evaluation |

The train/test generation or selection seeds are fixed. Cantilever uses
independent uniform random samples. Borehole and Piston use independently seeded
scrambled Latin hypercube samples. CCPP uses a seeded, non-overlapping selection
from the original UCI observations.

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

## 4. Combined Cycle Power Plant

Directory: `data/ccpp/`

The target is measured net hourly electrical power output. The empirical
surrogate equation being searched for has the form

$$
\widehat{PE} = f(AT, V, AP, RH).
$$

There is no known ground-truth expression. This problem therefore evaluates
predictive accuracy, complexity, interpretability, stability, and runtime, but
not exact equation recovery.

| Column | Symbol | Meaning | Observed source range |
|---|---:|---|---:|
| `ambient_temperature` | $AT$ | Ambient temperature (°C) | $[1.81, 37.11]$ |
| `exhaust_vacuum` | $V$ | Exhaust vacuum (cm Hg) | $[25.36, 81.56]$ |
| `ambient_pressure` | $AP$ | Ambient pressure (mbar) | $[992.89, 1033.30]$ |
| `relative_humidity` | $RH$ | Relative humidity (%) | $[25.56, 100.16]$ |
| `target` | $PE$ | Net electrical power output (MW) | $[420.26, 495.76]$ |

The source contains 9,568 complete observations. A fixed permutation with seed
20260815 selects 400 training rows and then 4,000 disjoint test rows without
changing their values. The normalized condition uses the documented full-data
ranges above, while the target is never scaled.

Source: UCI Combined Cycle Power Plant dataset, DOI
[`10.24432/C5002N`](https://doi.org/10.24432/C5002N), CC BY 4.0. The source CSV
is pinned by SHA-256 in the preparation script and benchmark configuration.

Preparation script: `surrogate/prepare_ccpp.py`

## 5. Naval propulsion simulator fuel flow

Directory: `data/naval_propulsion/`

The target is simulated gas-turbine fuel flow for a CODLAG frigate propulsion
plant. The empirical simulator surrogate being searched for is

$$
\widehat{m_f} = f(v,k_{Mc},k_{Mt}),
$$

where $v$ is ship speed, $k_{Mc}$ is compressor health, and $k_{Mt}$ is turbine
health. No analytical ground-truth equation is published.

| Column | Symbol | Meaning | Full simulator range |
|---|---:|---|---:|
| `ship_speed` | $v$ | Ship speed (knots) | $[3,27]$ |
| `compressor_decay` | $k_{Mc}$ | GT compressor decay coefficient | $[0.95,1]$ |
| `turbine_decay` | $k_{Mt}$ | GT turbine decay coefficient | $[0.975,1]$ |
| `target` | $m_f$ | Fuel flow (kg/s) | observed $[0.068,1.832]$ |

The 11,934 cases form the complete $9\times51\times26$ grid of the three
simulator controls. A permutation with seed 20260816 selects 400 training cases
and then 4,000 disjoint test cases without changing their values. Exact equation
recovery is unavailable; predictive accuracy, complexity, stability, and runtime
remain applicable.

Source: UCI Condition Based Maintenance of Naval Propulsion Plants, DOI
[`10.24432/C5K31K`](https://doi.org/10.24432/C5K31K). UCI currently marks the
dataset CC BY 4.0; the archived 2014 README also contains an older
non-commercial-use notice. Data are downloaded locally and are not committed.

Preparation script: `surrogate/prepare_naval_propulsion.py`

## 6. Light-aircraft wing weight

Directory: `data/wing_weight/`

The exact estimate being searched for is

$$
W = 0.036S_w^{0.758}W_{fw}^{0.0035}
\left(\frac{A}{\cos^2\Lambda}\right)^{0.6}q^{0.006}\lambda^{0.04}
\left(\frac{100t_c}{\cos\Lambda}\right)^{-0.3}
(N_zW_{dg})^{0.49}+S_wW_p.
$$

The implementation converts the published sweep angle from degrees to radians
before evaluating the cosine.

| Column | Symbol | Meaning | Range |
|---|---:|---|---:|
| `wing_area` | $S_w$ | Wing area (ft²) | $[150,200]$ |
| `fuel_weight` | $W_{fw}$ | Fuel weight in wing (lb) | $[220,300]$ |
| `aspect_ratio` | $A$ | Aspect ratio | $[6,10]$ |
| `sweep_angle_degrees` | $\Lambda$ | Quarter-chord sweep (degrees) | $[-10,10]$ |
| `dynamic_pressure` | $q$ | Cruise dynamic pressure (lb/ft²) | $[16,45]$ |
| `taper_ratio` | $\lambda$ | Taper ratio | $[0.5,1]$ |
| `thickness_chord_ratio` | $t_c$ | Aerofoil thickness/chord ratio | $[0.08,0.18]$ |
| `ultimate_load_factor` | $N_z$ | Ultimate load factor | $[2.5,6]$ |
| `design_gross_weight` | $W_{dg}$ | Design gross weight (lb) | $[1700,2500]$ |
| `paint_weight` | $W_p$ | Paint weight (lb/ft²) | $[0.025,0.08]$ |

This ten-dimensional benchmark tests fractional powers, weak variable effects,
a trigonometric term, disparate scales, and an additive contribution. Train and
test sets are independent scrambled Latin hypercubes using seeds 20260817 and
20260818.

Source: Forrester, Sóbester, and Keane (2008), *Engineering Design via
Surrogate Modelling*, DOI [`10.1002/9780470770801`](https://doi.org/10.1002/9780470770801).

Generator: `surrogate/generate_wing_weight.py`

## How results are judged

The current evaluator reports test-set RMSE, range-normalized RMSE, MAE, maximum
absolute error, $R^2$, runtime, and framework-independent SymPy expression
structure. For analytical problems it also tests exact algebraic agreement with
the equations above. A high $R^2$ alone does not prove equation recovery: two expressions may predict
similarly over these ranges while behaving differently elsewhere. Conversely,
an expression written in a different but algebraically equivalent form should
count as successful symbolic recovery. Predictive and symbolic-fidelity results
should therefore be reported separately in the thesis. For CCPP and Naval
Propulsion, exact-match fields remain unavailable rather than being counted as
failures.
