# Scientific compatibility and reproducibility

This package develops the notebooks accompanying Muhammad Izzatullah, Isa Eren
Yildirim, Umair Bin Waheed and Tariq Alkhalifah (2022), *Laplace HypoPINN:
physics-informed neural network for hypocenter localization and its predictive
uncertainty*, Machine Learning: Science and Technology **3**, 045001,
[doi:10.1088/2632-2153/ac94b3](https://doi.org/10.1088/2632-2153/ac94b3).
The original scientific attribution and archived source comments are retained.
The maintained code is licensed under [MIT](../LICENSE); [third-party notices](../THIRD_PARTY_NOTICES.md)
define the exceptions for archived code and bundled data. The article's CC BY 4.0
terms apply separately to the publication.

The [archived notebooks](../archive/notebooks/) are historical evidence. Cell
references below use **zero-based indices in the notebook JSON**, including
Markdown cells. Numerical settings below define compatibility targets; they are
not claims that the paper's tables, timing measurements or uncertainty calibration
have been reproduced. Consult recorded validation artifacts for completed runs.

## Presets

All historical presets use float32, seeds 4321 for NumPy and PyTorch, a fixed
collocation subset sampled once without replacement, and 3,001 Adam updates.
Adam uses learning rate 0.005, betas (0.5, 0.9), epsilon 1e-8 and zero weight decay.
A cosine scheduler steps after each optimizer update with `T_max=3001`,
`eta_min=0`. The final iterate is the sampling center.

The notebook reproduction model is `2 → 16 → 16 → 32 → 32 → 16 → 16 → 1`,
with Mish hidden activations, a linear output and 2,737 parameters. It is a preset
and regression fixture: depth, widths and activation are configurable, and custom
differentiable pointwise scalar models can supply a functional callable or an
`nn.Module` adapter. Generic Jacobian methods support custom models; jet-based
methods require a compatible declared adapter. Compatibility with the notebook
architecture is not a restriction on research models.

| Notebook / preset | Grid (z rows × x columns) | True source (z,x), km | Collocation count | Surface receiver columns | Initialization | Posterior add / multiply |
| --- | --- | --- | --- | --- | --- | --- |
| `EX_1` | 101 × 151 | (1,1.5) | 2,501 | 0,15,…,150 | Four sequential variants below | No posterior experiment |
| `EX_2` | 101 × 151 | (1,1.5) | 2,501 | 0,15,…,150 | Kaiming normal | 1000 / .001 |
| `EX_3` | 246 × 305 | (.7,1) | 5,001 | 0,60,…,300 | Kaiming normal | 5000 / .001 |
| `EX_3_2` | 246 × 305 | (.75,1.1) | 5,001 | 0,60,…,300 | Kaiming normal | 5000 / .001 |
| `EX_3_3` | 246 × 305 | (.65,.9) | 5,001 | 0,60,…,300 | Kaiming normal | 5000 / .001 |

`EX_1` executes Kaiming normal, Xavier normal, Kaiming uniform, Xavier uniform
in that order without reseeding each network. Biases are zero. Weight
initializers use the PyTorch defaults (Kaiming: `a=0`, `mode='fan_in'`,
`nonlinearity='leaky_relu'`; Xavier: `gain=1`). Every `nn.Linear` constructor
consumes its default draws before the explicit reinitialization; preserving only
the final initializer name is insufficient for random-stream parity.

`EX_2` and Otway presets select `legacy_diagonal`: 1,000 coordinate batches of
the listed collocation count, sampled with replacement at the final parameters,
followed by 1,000 independent parameter draws. Receiver data stay fixed across
curvature batches. The precision is `add + multiply * curvature`; the standard
deviation is its inverse square root. Each draw perturbs the same final parameter
center rather than the preceding draw. See [methods](methods.md) for the precise
statistic and corrected alternatives.

## Coordinates, data and random streams

Coordinates and locations are `(z,x)` in km, velocity is km/s and travel time is
seconds. Meshes use `ij` indexing and row-major flattening, with x varying fastest.
Localization returns the first minimum in that order, using actual axis values;
MAP location, posterior mean location and population standard deviation (`ddof=0`)
are distinct quantities. Some historical plotting tuples reverse the axes to
(x,z); this does not change the numerical coordinate convention.

Gradient examples use z=0…2 and x=0…3 at .02 km spacing, with
`v(z,x)=2+.5*z`. The reference field is homogeneous distance/velocity when the
velocity gradient is zero, otherwise the analytic constant-gradient solution
used in `EX_2` cell 12. Receiver indices select the surface row.

Otway reads the root [otway2.csv](../otway2.csv) without a header and transposes
it to 246 × 305. Supply an explicit velocity-data path for Otway runs; relative
paths resolve against the configuration file. The wheel does not bundle a second
copy of the dataset. Notebook cell 11 constructs bounds from floating-point
`arange` endpoints, then uses `linspace` over the entire velocity shape. In the
reference environment, z spans 0…1.28 and x spans 0…1.704; actual spacings are
1.28/245 and 1.704/304 km, respectively, rather than .008 km. Compatibility
presets preserve these effective bounds. The fast-marching level set is +1
except −1 at `(int(sz // dz), int(sx // dx))`; pass both axis spacings to
`skfmm.travel_time` (cell 13). This finite level-set construction is part of the
reference, not an exact point-source boundary condition.

Local random streams must preserve both initialization draws and NumPy subset
selection order without changing caller RNG state. Legacy posterior noise is drawn
once per layer as a combined weight/bias matrix, rather than as separate tensors.
The canonical corrected parameter layout is row-major `[W|b]` by layer; legacy
serialization requires an explicit mapping. Backend and version differences may
require numerical tolerances. Bitwise equality across CPU, CUDA or PyTorch versions
is not promised. CPU is the initial verification target; explicitly select a
supported device instead of inheriting the notebooks' hardcoded GPU index 1.

Synthetic true-source coordinates generate reference receiver times and permit
error evaluation. Inversion from supplied receiver coordinates and observed times
does not require a true source or reference travel-time field.

## Paper and notebook differences

| Topic | Paper reference | Archived code and maintained convention |
| --- | --- | --- |
| Architecture | §3 describes five hidden layers | All numerical notebooks, cell 6: six hidden layers; the paper §3.2 count of 2,737 parameters matches this code |
| Counts | §§3.1–3.2: 2,500/5,000 collocation points, 3,000 epochs | `EX_2` cells 10/18; Otway cells 11/20: 2,501/5,001 points and 3,001 updates |
| Positivity | Equation (2) contains PDE and receiver losses | Cell 8 also adds `mean(((1-sign(T))*abs(T))**2)` |
| Training prior | §§2.2, 3.1–3.2 describe a Gaussian prior / Tikhonov term | Cell 8 and training cells contain no parameter L2 term or Adam weight decay; default training prior precision is zero |
| Otway axes | §3.2 states .008 km spacing with a 246 × 305 shape | Cells 10–13 produce the effective axes described above |
| Curvature | §2.2 discusses Fisher / Hessian approximations | `EX_2` cell 30 and Otway cell 31 accumulate squared gradients of total batch-mean loss, not per-observation Fisher or exact Hessian |
| Covariance sign | Equation (7) notation can imply a negative covariance | Always use positive precision and its inverse; never a negative covariance |

Counts, architecture, positivity weight, training L2 precision and grid axes are
explicit configuration choices. Deviations are custom variants. There is no single
“paper-corrected” preset: the article does not specify a unique prior strength and
its Otway dimensions and stated spacing do not agree.

The archived curvature loops instantiate EFB inside each batch and rely on hooks
that cannot correctly represent this derivative loss. Historical KFAC/EFB/INF
outputs are not mathematical acceptance references for maintained methods.
`EX_3_plot` contains hardcoded inferred coordinates; maintained plots must use
saved result artifacts instead.

## Evidence required for a reproduction claim

A short run establishes execution, not reproduction. Full `EX_2` comparison needs
3,001 updates, 1,000 curvature batches and 1,000 posterior draws, with separate
training, curvature and prediction comparisons against the legacy reference in
the same environment. Full `EX_1` and Otway outcomes require separate recorded
runs. Published timing values are not speed targets or local measurements.

Record resolved configuration, method conventions, seed/RNG state, source revision
and dirty status, dependencies, dtype/device, input checksum, timings, training
history, final parameters, MAP location and posterior locations. Save fields only
when requested. Archive byte checksums establish preservation, not numerical
agreement. Report any missing or incomplete validation stage explicitly.

The reported uncertainty measures network-weight-induced predictive variation.
It does not by itself establish calibrated velocity-model or measurement-noise
uncertainty, or validate a physical source posterior.
