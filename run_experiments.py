"""Build and execute the paper's comparison/ablation/top-k experiment matrix."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from train import ALGORITHMS

ROOT = Path(__file__).resolve().parent


def make_plan(template, suite="all", seeds=(42, 43, 44), dimensions=(9, 18, 27, 36), smoke=False, algorithms=ALGORITHMS):
    variants = []
    if suite in ("all", "comparison"):
        variants += [(algorithm, dim, "none", 2) for algorithm in algorithms for dim in dimensions]
    if suite in ("all", "ablation"):
        variants += [("dhea_mecd", dim, ablation, 2) for dim in (18, 36)
                     for ablation in ("none", "no_encoder_attention", "no_mecd")]
    if suite in ("all", "topk"):
        variants += [("dhea_mecd", 36, "none", k) for k in range(1, 6)]
    unique = sorted(set(variants))
    result = []
    for algorithm, dim, ablation, k in unique:
        for seed in seeds:
            config = deepcopy(template)
            config.update(algorithm=algorithm, ablation=ablation)
            config["environment"]["observation_dim"] = dim
            config["training"]["seed"] = seed
            config["agent"]["top_k"] = k
            if smoke:
                config["protocol"] = "SMOKE_FUNCTIONAL_TEST_NOT_PAPER_RESULTS"
                config["environment"]["max_step"] = 8
                config["training"].update(max_steps=8, warmup=4)
                config["agent"].update(latent_dim=16, token_dim=24, encoder_hidden=32, gru_hidden=32,
                                      batch_size=4, capacity=32, sequence_length=2)
                config["baseline"].update(rollout_steps=4, epochs=1, cg_steps=2, line_search_steps=3)
            run_id = f"{algorithm}_d{dim}_{ablation}_k{k}_seed{seed}"
            digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            result.append({"run_id": run_id, "config": config, "sha256": digest,
                           "training_status": "pending", "evaluation_status": "pending"})
    return result


def write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "experiments/paper.json")
    parser.add_argument("--suite", choices=["all", "comparison", "ablation", "topk"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--dimensions", type=int, nargs="+", choices=[9, 18, 27, 36], default=[9, 18, 27, 36])
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=1_000_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/paper"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.episodes, args.eval_episodes) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Episode counts must be positive and seeds distinct")
    episodes, eval_episodes = (1, 2) if args.smoke else (args.episodes, args.eval_episodes)
    planned = make_plan(json.loads(args.template.read_text()), args.suite, args.seeds, args.dimensions, args.smoke, args.algorithms)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    manifest = {"protocol": "smoke" if args.smoke else "paper_configuration",
                "episodes": episodes, "eval_episodes": eval_episodes,
                "evaluation_seed": args.evaluation_seed, "device": args.device, "runs": planned}
    if path.exists():
        previous = json.loads(path.read_text())
        signature = lambda x: [(r["run_id"], r["sha256"]) for r in x["runs"]]
        if signature(previous) != signature(manifest) or any(previous[k] != manifest[k] for k in
                ("episodes", "eval_episodes", "evaluation_seed", "device")):
            raise ValueError("Output directory contains a different plan; choose a fresh directory")
        manifest = previous
    write_manifest(path, manifest)
    print(f"{len(planned)} runs; {episodes} train episodes/run; {eval_episodes} held-out episodes/run", flush=True)
    if args.plan_only:
        return
    for number, run in enumerate(manifest["runs"], start=1):
        directory = args.output / run["run_id"]
        directory.mkdir(exist_ok=True)
        config_path = directory / "config.json"
        config_path.write_text(json.dumps(run["config"], indent=2))
        print(f"[{number}/{len(planned)}] {run['run_id']}", flush=True)
        if run["training_status"] == "complete" and not (directory / "latest.pt").exists():
            raise FileNotFoundError(f"Completed run is missing its checkpoint: {directory}")
        if run["evaluation_status"] == "complete" and not (directory / "evaluation/summary.json").exists():
            raise FileNotFoundError(f"Completed run is missing evaluation outputs: {directory}")
        if run["training_status"] != "complete":
            checkpoint = directory / "latest.pt"
            remaining = episodes
            command = [sys.executable, str(ROOT / "train.py"), "--config", str(config_path),
                       "--output", str(directory), "--device", args.device]
            if checkpoint.exists():
                import torch
                completed = int(torch.load(checkpoint, map_location="cpu", weights_only=False)["episode"]) + 1
                remaining -= completed
                command += ["--resume", str(checkpoint)]
                # Drop uncheckpointed CSV rows after interruption before resuming.
                import csv
                metrics = directory / "training.csv"
                if metrics.exists():
                    with metrics.open() as file:
                        reader = csv.DictReader(file)
                        fields, rows = reader.fieldnames, list(reader)
                    with metrics.open("w", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(r for r in rows if int(r["episode"]) < completed)
            elif (directory / "training.csv").exists():
                # No committed model exists, so restart this incomplete run.
                (directory / "training.csv").unlink()
            if remaining > 0:
                command += ["--episodes", str(remaining)]
                with (directory / "training.log").open("a") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            run["training_status"] = "complete"
            write_manifest(path, manifest)
        if run["evaluation_status"] != "complete":
            evaluation = directory / "evaluation"
            # Evaluation is deterministic with a fixed checkpoint/seed; restart interrupted output.
            if (evaluation / "evaluation.csv").exists():
                (evaluation / "evaluation.csv").unlink()
            command = [sys.executable, str(ROOT / "train.py"), "--evaluate", str(directory / "latest.pt"),
                       "--episodes", str(eval_episodes), "--evaluation-seed", str(args.evaluation_seed),
                       "--device", args.device, "--output", str(evaluation), "--save-trajectories"]
            with (directory / "evaluation.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            run["evaluation_status"] = "complete"
            write_manifest(path, manifest)
    print(f"Complete: {path}", flush=True)


if __name__ == "__main__":
    main()
