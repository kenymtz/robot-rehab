"""Apply the decision rule and pack everything to take back."""
import platform
import shutil
import zipfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from ..pipeline import Context, read_json, write_json
from ..report import build_summary, plot_recovery, render_markdown
from ..video import make_video
from .final import case_name

PACKAGES = ["gymnasium", "mujoco", "numpy", "stable-baselines3", "torch"]


def build_manifest(ctx: Context) -> dict:
    stages = {path.parent.name: read_json(path)["fingerprint"] for path in ctx.root.glob("*/done.json")}
    return {"config": ctx.cfg, "date": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "workers": ctx.workers,
            "packages": {name: version(name) for name in PACKAGES}, "stages": stages}


def _make_median_video(ctx: Context, summary: dict) -> bool:
    """Show the median case. Without one, show the first case, so this path is always exercised."""
    case = summary["median_case"] or {"robot": ctx.cfg["robots"][0], "pattern": ctx.cfg["final"]["patterns"][0]}
    robot, pattern = case["robot"], case["pattern"]
    traj = ctx.stage_dir("final") / "traj"
    paths = [traj / f"{case_name(robot, -1, 'healthy', 'large')}.npz",
             traj / f"{case_name(robot, pattern, 'none', 'large')}.npz",
             traj / f"{case_name(robot, pattern, 'mutation', 'large')}.npz"]
    return make_video(paths, ctx.out / "comparacion.mp4", ctx.cfg["env_id"], ctx.log)


def _zip(ctx: Context, target: Path) -> None:
    extras = [ctx.stage_dir("gate") / "gate.json", ctx.stage_dir("damage_curve") / "curve.json",
              ctx.stage_dir("damage_curve") / "damage_level.json", ctx.stage_dir("tune") / "tuned.json",
              ctx.stage_dir("final") / "final.json"]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in ctx.out.iterdir():
            if path.is_file() and not path.name.startswith("."):
                archive.write(path, path.name)
        for path in extras:
            archive.write(path, path.name)
        for path in (ctx.stage_dir("final") / "traj").glob("*.npz"):
            archive.write(path, f"traj/{path.name}")


def run(ctx: Context) -> None:
    records = read_json(ctx.stage_dir("final") / "final.json")
    level = read_json(ctx.stage_dir("damage_curve") / "damage_level.json")
    summary = build_summary(records, ctx.cfg["report"]["bootstrap"])
    video_ok = _make_median_video(ctx, summary)
    meta = {"smoke": ctx.cfg["smoke"], "level": level, "video": video_ok}
    (ctx.out / "report.md").write_text(render_markdown(summary, meta), encoding="utf-8")
    plot_recovery(summary, ctx.out / "recuperacion.png")
    write_json(ctx.out / "summary.json", summary)
    write_json(ctx.out / "manifest.json", build_manifest(ctx))
    target = ctx.root.parent.parent / ctx.cfg["report"]["zip_name"]
    temporary = target.with_name(target.name + ".part")
    _zip(ctx, temporary)
    shutil.move(temporary, target)
    ctx.log(f"[report] veredicto: {summary['verdict']}. Resultados en {target}")
