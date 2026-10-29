"""Render evaluation traces; isolate native OpenGL failures from the caller."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def pyvista_render(trace, output, interactive):
    import pyvista as pv
    auv, targets = trace["auv"], trace["targets"]
    plotter = pv.Plotter(off_screen=not interactive, window_size=(1200, 900))
    plotter.set_background("white")
    if len(auv)>1:
        plotter.add_mesh(pv.lines_from_points(auv), color="goldenrod", line_width=4)
    plotter.add_mesh(pv.Sphere(radius=float(trace.get("agent_radius", 0.025)), center=auv[-1]), color="yellow")
    for i in range(targets.shape[1]):
        if len(targets)>1:
            plotter.add_mesh(pv.lines_from_points(targets[:, i]), color="red", line_width=2)
        plotter.add_mesh(pv.Sphere(radius=float(trace.get("target_radius", 0.1)), center=targets[-1, i]), color="red")
    for point in trace["obstacles"][-1]:
        plotter.add_mesh(pv.Sphere(radius=float(trace.get("obstacle_radius", 0.08)), center=point), color="gray", opacity=0.5)
    plotter.show_grid()
    plotter.add_axes()
    plotter.view_isometric()
    plotter.show(screenshot=str(output), auto_close=True)


def matplotlib_render(trace, output, interactive):
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    auv, targets = trace["auv"], trace["targets"]
    fig = plt.figure(figsize=(7, 6), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*auv.T, color="goldenrod", label="AUV")
    ax.scatter(*auv[-1], color="yellow", edgecolors="black", s=45)
    for i in range(targets.shape[1]):
        ax.plot(*targets[:, i].T, color="red", label="Targets" if i == 0 else None)
        ax.scatter(*targets[-1, i], color="red", s=25)
    obstacles = trace["obstacles"][-1]
    if len(obstacles):
        ax.scatter(*obstacles.T, color="gray", alpha=0.5, label="Obstacles")
    bound = float(trace.get("field_size", 1.0))
    ax.set(xlabel="x (simulation units)", ylabel="y (simulation units)", zlabel="z (simulation units)",
           xlim=(-bound,bound), ylim=(-bound,bound), zlim=(-bound,bound),
           title="Trajectory preview (Matplotlib renderer)")
    ax.set_box_aspect((1,1,1))
    ax.legend()
    fig.savefig(output, dpi=300)
    if interactive:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, default=Path("trajectory.png"))
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--backend", choices=["pyvista", "matplotlib"], default="pyvista")
    parser.add_argument("--strict-pyvista", action="store_true", help="Fail if PyVista cannot render; do not create a fallback preview")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dhea-matplotlib"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args._worker:
        with np.load(args.trace) as trace:
            pyvista_render(trace, args.output, args.interactive)
        return
    renderer, error = args.backend, None
    if renderer == "pyvista":
        command = [sys.executable, str(Path(__file__).resolve()), str(args.trace.resolve()),
                   "--output", str(args.output.resolve()), "--_worker"]
        if args.interactive:
            command += ["--interactive"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            error = f"PyVista process exit {result.returncode}: " + result.stderr[-2500:]
            if args.strict_pyvista:
                raise RuntimeError(error)
            print("PyVista could not initialize its renderer; creating an explicitly labeled Matplotlib preview.", file=sys.stderr)
            renderer = "matplotlib"
    if renderer == "matplotlib":
        with np.load(args.trace) as trace:
            matplotlib_render(trace, args.output, args.interactive)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(
        {"renderer": renderer, "fallback_reason": error, "source": args.trace.name,
         "units": "simulation units; no implicit metre calibration"}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
