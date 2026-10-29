# Author-provided appendix alignment

The authoritative dimensional specification is the author-provided `1015appendix.pdf`, two pages, received and checked on 2026-10-29. Both pages were extracted and visually inspected. Its feature identities and one-based index ranges agree with the previously available local dimensional specification documents.

No observation dimension, feature order, model input shape or existing checkpoint tensor shape changed during this reconciliation. The source designation, README and regression checks were updated. This audit establishes schema alignment, not numerical reproduction of the manuscript's experimental results.

## One-based index audit

| Schema / source page | Indices and specified features | Implementation |
| --- | --- | --- |
| 9D / page 1 | 1–3 local relative displacement; 4 Euclidean distance; 5 capture indicator; 6–8 linear velocity; 9 normalized time | `appendix_observation`, 9D branch; local displacement rotated by inverse AUV yaw/pitch |
| 18D / page 1 | 1–3 AUV global position; 4–6 target-cluster centroid; 7 distance; 8 capture; 9–11 velocity; 12–13 yaw sin/cos; 14–15 pitch sin/cos; 16 time; 17–18 surface/seabed distances | 18D branch; explicit top/bottom workspace distances occupy the final two slots |
| 27D / page 1 | 1–3 AUV global position; 4–6 target-cluster centroid; 7–8 distance/capture; 9–11 velocity; 12–15 yaw/pitch encodings; 16 time; 17–19 current; 20–22 nearest obstacle 1 offset; 23 radius; 24–26 nearest obstacle 2 offset; 27 radius | 27D branch; obstacles sorted by current distance, with relative offsets and radius in each slot |
| 36D / page 2 | 1–3 AUV global position; 4–6 target-cluster centroid; 7–9 guidance vector; 10 distance; 11 capture; 12–14 velocity; 15–16 yaw sin/cos; 17–18 pitch sin/cos; 19 yaw rate; 20 time; 21–23 current; 24–27 obstacle 1 offset/radius; 28–31 obstacle 2 offset/radius; 32 vehicle noise; 33 biological noise; 34 geological noise; 35 turbulence noise; 36 perception confidence | 36D branch; named four-noise order and final confidence slot verified |

The programmatic slot names are in `experiments/features.json` and `hyenv/observations.py`. The manuscript's semantic decomposition is implemented with index gathers rather than slicing a full observation arbitrarily. The routing sizes are respectively (5,4,0,0), (8,8,2,0), (8,8,11,0), and (11,9,11,5) for space/motion/barrier/noise. This routing assignment is an implementation of the main manuscript's four branches; the appendix itself only defines the feature list and order.

## Regression checks

- Analytic 9D scene: with yaw = pi/2 and pitch = 0, world +x displacement maps to body -y. Velocity and time occupy their declared slots, and the capture flag changes when range is crossed.
- Analytic 18D scene: yaw/pitch sin/cos and normalized surface/seabed distances are checked at a known position and orientation.
- 27D scene: local current, obstacle offset, physical radius and the missing second-obstacle slot occupy the declared indices.
- 36D scene: target centroid, relative guidance, nearest-obstacle ordering, yaw rate, time, all four nonzero noise slots and confidence are checked independently of the network.
- All four schemas exercise the semantic encoders; absent branches have no data-dependent gradients. The existing training/checkpoint/baseline tests remain applicable.

## Definitions that remain implementation choices

The new appendix does not supply additional equations resolving the following. They remain disclosed choices rather than author-specified mathematics:

1. Which targets belong to the centroid after acquisition. The code uses pending targets, falling back to all targets after completion.
2. Whether 36D guidance has unit length. The code uses the world-frame displacement to the centroid, scaled to [0,1], without an additional unit-vector normalization.
3. The confidence formula. The code uses `1 / (1 + mean(abs(noise)) / noise_std)`, with a small denominator guard when the noise scale is zero.
4. Whether each noise-intensity feature denotes a raw process sample, absolute amplitude or windowed power. The code uses samples of the four processes defined in the manuscript and applies a bounded arctan normalization.
5. Exact scaling constants, the world-frame convention for velocity, handling fewer than two obstacles, task-specific capture rules, simulator calibration, and hyperparameters absent from the main manuscript.

These ambiguities do not change the verified slot count or order. The appendix also does not resolve the main manuscript's hard-versus-soft target-update description or provide the physical metre calibration needed to reproduce its trajectory-length table.
