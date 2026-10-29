"""Aggregate actual experiment logs; independent training seeds define CI n."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

KEYS = ("algorithm", "dimension", "ablation", "top_k")


def read_csv(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and not fields:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Statistics require nonempty finite observations")
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "sd": None, "ci_low": None, "ci_high": None, "n_seeds": len(values)}
    sd = float(values.std(ddof=1))
    half = float(t.ppf(0.975, len(values)-1)*sd/np.sqrt(len(values)))
    return {"mean": mean, "sd": sd, "ci_low": mean-half, "ci_high": mean+half, "n_seeds": len(values)}


def truth(value):
    return value in (True, "True", "true", "1")


def aggregate(root, allow_incomplete=False, length_bins=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    seeds_rows, curves, distributions, excluded = [], defaultdict(list), [], []
    energy_samples = defaultdict(list)
    for run in manifest["runs"]:
        directory = root / run["run_id"]
        if run["evaluation_status"] != "complete" or run["training_status"] != "complete":
            excluded.append(run["run_id"])
            continue
        cfg = run["config"]
        metadata = {"algorithm": cfg["algorithm"], "dimension": cfg["environment"]["observation_dim"],
                    "ablation": cfg["ablation"], "top_k": cfg["agent"]["top_k"]}
        key = tuple(metadata.values())
        rows = read_csv(directory / "evaluation/evaluation.csv")
        training = read_csv(directory / "training.csv")
        if len(rows) != manifest["eval_episodes"] or len(training) != manifest["episodes"]:
            raise ValueError(f"Log budget mismatch in {run['run_id']}")
        if len({r["episode"] for r in rows}) != len(rows):
            raise ValueError("Duplicate evaluation episodes")
        if len({r["episode"] for r in training}) != len(training):
            raise ValueError("Duplicate training episodes")
        seed = cfg["training"]["seed"]
        successful = [r for r in rows if truth(r["success"])]
        per_seed = {**metadata, "seed": seed, "eval_episodes": len(rows),
                    "success_rate": float(np.mean([truth(r["success"]) for r in rows])),
                    "collision_rate": float(np.mean([truth(r["collision"]) for r in rows])),
                    "reward": float(np.mean([float(r["reward"]) for r in rows])),
                    "energy_cost": float(np.mean([float(r["energy_cost"]) for r in rows])),
                    "trajectory_length": float(np.mean([float(r["trajectory_length"]) for r in rows])),
                    "successful_trajectory_length": float(np.mean([float(r["trajectory_length"]) for r in successful])) if successful else None,
                    "inference_mean_ms": float(np.mean([float(r["inference_mean_ms"]) for r in rows])),
                    "mean_episode_p95_ms": float(np.mean([float(r["inference_p95_ms"]) for r in rows]))}
        seeds_rows.append(per_seed)
        energy_samples[key].extend(float(r["energy_cost"]) for r in rows)
        for row in training:
            curves[(*key, int(row["episode"]))].append(float(row["reward"]))
        unit = "m" if cfg["environment"].get("length_scale_m") else "simulation_unit"
        bins = length_bins if length_bins is not None else ([200, 300, 350, 400, 500] if unit == "m" else None)
        if bins is not None:
            lengths = np.array([float(r["trajectory_length"]) for r in successful])
            counts, _ = np.histogram(lengths, bins=bins)
            labels = [(f"below_{bins[0]}", int(np.sum(lengths < bins[0])))]
            labels += [(f"{left}_{right}", int(count)) for left, right, count in zip(bins[:-1], bins[1:], counts)]
            labels += [(f"above_{bins[-1]}", int(np.sum(lengths > bins[-1])))]
            for label, count in labels:
                distributions.append({**metadata, "seed": seed, "length_unit": unit, "bin": label,
                                      "count": count, "successful_episodes": len(successful),
                                      "fraction": count/len(successful) if successful else None})
    if excluded and not allow_incomplete:
        raise ValueError(f"{len(excluded)} runs incomplete; finish them or use --allow-incomplete")
    grouped = defaultdict(list)
    for row in seeds_rows:
        grouped[tuple(row[k] for k in KEYS)].append(row)
    summaries = []
    metrics = ("success_rate", "collision_rate", "reward", "energy_cost", "trajectory_length",
               "successful_trajectory_length", "inference_mean_ms", "mean_episode_p95_ms")
    for key, rows in grouped.items():
        for metric in metrics:
            values = [r[metric] for r in rows if r[metric] is not None]
            if values:
                summaries.append({**dict(zip(KEYS, key)), "metric": metric, **interval(values),
                                  "n_eval_episodes": sum(r["eval_episodes"] for r in rows)})
    curve_rows = [{**dict(zip(KEYS, key[:4])), "episode": key[4], **interval(values)}
                  for key, values in sorted(curves.items())]
    # Pair each method with PPO using the same training seed and held-out scenarios.
    reference = {(r["dimension"], r["seed"]): r["energy_cost"] for r in seeds_rows if r["algorithm"] == "ppo"}
    savings = []
    for row in seeds_rows:
        denominator = reference.get((row["dimension"], row["seed"]))
        if denominator is not None and denominator > 0:
            savings.append({**{k: row[k] for k in (*KEYS, "seed")}, "reference": "ppo",
                            "saving_percent": 100*(1-row["energy_cost"]/denominator)})
    episode_sd = [{**dict(zip(KEYS, key)), "mean_over_eval_episodes": float(np.mean(values)),
                   "sd_over_eval_episodes": float(np.std(values, ddof=1)) if len(values)>1 else None,
                   "n_eval_episodes": len(values)} for key, values in energy_samples.items()]
    return manifest, seeds_rows, summaries, curve_rows, distributions, savings, episode_sd, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--length-bins", type=float, nargs="+")
    args = parser.parse_args()
    if args.length_bins and (len(args.length_bins)<2 or np.any(np.diff(args.length_bins)<=0)):
        parser.error("length bins must be strictly increasing")
    result = aggregate(args.root, args.allow_incomplete, args.length_bins)
    manifest, seeds, summary, curves, bins, savings, energy_sd, excluded = result
    output = args.output or args.root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("per_seed.csv", seeds), ("summary.csv", summary), ("learning_curves.csv", curves),
                           ("trajectory_distribution.csv", bins), ("energy_savings_vs_ppo.csv", savings),
                           ("energy_episode_sd.csv", energy_sd)):
        write_csv(output / filename, rows)
    notes = {"protocol": manifest["protocol"], "independent_unit": "training seed",
             "confidence_interval": "two-sided 95% Student t over per-seed means, n-1 df; absent when n<2",
             "learning_curve": "episode return; no smoothing, interpolation, missing-seed imputation, or clipping of CI",
             "energy_episode_sd": "descriptive SD over evaluation episodes, not a seed-level confidence interval",
             "successful_trajectory": "conditional on success; no successes yields missing, never zero",
             "length_bins": "paper metre bins only when length_scale_m is explicitly configured; out-of-range successes retained",
             "exclusions": excluded, "generated_tables": ["per_seed", "summary", "learning_curves", "energy_savings", "energy_episode_sd"]}
    (output / "statistics.json").write_text(json.dumps(notes, indent=2))
    print(f"Summarized {len(seeds)} trained-seed runs into {output}")


if __name__ == "__main__":
    main()
