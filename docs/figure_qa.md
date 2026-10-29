# Figure QA record

Validated on the functional smoke logs, not on formal paper experiments. The 40-configuration matrix contains one trained seed per configuration; a separate 36D run contains three independent seeds. Every rendered smoke figure carries an explicit functional-test title.

Source preflight: 19 PASS, 1 WARN, 0 FAIL. The sole warning is that PNG previews are exported instead of submission TIFF rasters; editable SVG/PDF are provided, and no TIFF submission artifact is claimed. PNG resolution is 600 dpi. All exported comparison/ablation/top-k/paired-saving PDFs, plus the three-seed PDFs, passed the 5 pt glyph audit; the minimum observed was 6 pt.

| Figure/panels inspected | Quantity and replicate unit | Uncertainty | Labels / collision / interpretation |
| --- | --- | --- | --- |
| Convergence, 9D | Observed episode return, training seed | None for n=1 | One actual episode shown as a point, no invented time course; overlapping returns preserved |
| Convergence, 18D | Observed episode return, training seed | None for n=1 | Same conventions; algorithm names and dimension readable |
| Convergence, 27D | Observed episode return, training seed | None for n=1 | Same conventions; overlapping points are actual equal/near-equal returns |
| Convergence, 36D | Observed episode return, training seed | None for n=1 | Same conventions; no convergence conclusion from a single episode |
| Success, 9D/18D/27D/36D | Per-seed evaluation success rate | None for n=1 | All zero values preserved; labels clear, no performance claim |
| Collision, 9D/18D/27D/36D | Per-seed evaluation collision rate | None for n=1 | All zero values preserved; short episode budget does not establish safety |
| Energy, 9D/18D/27D/36D | Mean normalized command cost per seed | None for n=1 | Values traceable to CSV; no physical battery-energy claim |
| Latency, 9D/18D/27D/36D | Per-seed mean decision time, local CPU | None for n=1 | Explicit ms label; smoke CPU measurements do not reproduce RTX 4090 timing |
| Module ablation, 18D | Episode return for three model variants | None for n=1 | Equal initial returns overlap intentionally; no selective jitter or exclusion |
| Module ablation, 36D | Episode return for three model variants | None for n=1 | Same conventions |
| Top-k, success panel | Per-seed success rate, k=1…5 | None for n=1 | Five k values retained, no superiority inference |
| Top-k, collision panel | Per-seed collision rate, k=1…5 | None for n=1 | Same conventions |
| Saving vs PPO, 9D/18D/27D/36D | Matched-seed cost ratio, Eq. (43) | None for n=1 | Extreme negative ratios from near-zero untrained PPO costs are retained, not clipped; smoke ratios are not scientific results |
| Three-seed 36D return | Mean return across three independent training seeds | Two-sided 95% Student t CI | Mean point and interval visible; no smoothing or episode imputation |
| Three-seed 36D success/collision/energy/latency | Mean across per-seed evaluation means | Two-sided 95% Student t CI | Same statistical unit and interval convention across panels |

Matplotlib three-dimensional trajectory preview rendered and inspected. The PyVista worker could not initialize OpenGL in this session; the parent survived the native failure and generated a preview explicitly labeled Matplotlib, with a renderer provenance JSON. Native PyVista rendering remains unverified here.

The scripts must be rendered and visually checked again against the eventual formal experiment logs: long learning curves, nonzero success distributions and multi-seed uncertainty have different geometric demands from these smoke artifacts. The smoke QA does not certify a final submission figure.
