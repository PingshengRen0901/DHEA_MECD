# Experiment figure contract

- Purpose: compare observed learning and held-out behavior under a shared simulator. No claim of method superiority is made before actual runs exist. Smoke figures demonstrate data plumbing only.
- Evidence: return curves address convergence; success and collision panels address acquisition and safety; latency and normalized command energy address computational/actuation cost. Ablations and k sweeps address components and routing breadth.
- Layout: quantitative grid, one axis per observation dimension; separate, clearly labeled comparison, ablation and k-sweep figures. DHEA-MECD has the strongest line weight, with consistent baseline colors.
- Backend: Python/matplotlib, matching the repository's language-specific workflow; project preference persisted in experiments/figure_backend.json.
- Export: approximately 183 mm wide comparison grids; 7 pt text, no glyph below 5 pt; editable SVG/PDF and 600 dpi PNG; CSV source data beside figures.
- Statistics: independent unit is a training seed. Curves and metric intervals use 95% Student t intervals across seed means. One seed has no CI. No synthetic values, smoothing, clipping of uncertainty, or convenient episode exclusions. Descriptive energy SD is exported separately.
- Review risks: a three-seed t interval is unstable; seed CI does not capture simulator misspecification. Path lengths require explicit physical calibration to be called metres. All action adapters and unresolved manuscript parameters must be disclosed.
