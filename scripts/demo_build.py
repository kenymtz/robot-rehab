"""Put the demo's data and clips and the official results into one page: runs/demo/site/index.html.

Run after scripts/demo_render.py, the official run and scripts/offline_error.py.
Usage: uv run --frozen python scripts/demo_build.py
"""
import base64
import json
from pathlib import Path

from robotrehab.evaluate import recovery

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "runs" / "demo" / "site"
RUN = ROOT / "runs" / "full"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def video_uri(path: Path) -> str:
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()


def official() -> dict:
    summary = read(RUN / "report" / "summary.json")
    config = read(RUN / "report" / "manifest.json")["config"]
    records = read(RUN / "final" / "final.json")
    episodes = config["final"]["episodes"]

    def tally(condition: str, memory: str) -> dict:
        rows = [r for r in records if r["condition"] == condition and r["memory"] == memory]
        return {"survivors": sum(r["survivors"] for r in rows), "episodes": episodes * len(rows)}

    healthy = {r["robot"]: r["return"] for r in records if r["condition"] == "healthy"}
    damaged = {(r["robot"], r["pattern"]): r["return"] for r in records if r["condition"] == "none"}
    cases = [{"robot": r["robot"], "pattern": r["pattern"],
              "recovery": recovery(r["return"], damaged[r["robot"], r["pattern"]], healthy[r["robot"]])}
             for r in records if r["condition"] == "mutation" and r["memory"] == "large"]
    conditions = []
    for key, stats in summary["conditions"].items():
        method, memory = key.split("/")
        conditions.append({"method": method, "memory": memory, "median": stats["median"], "ci_low": stats["ci_low"],
                           "ci_high": stats["ci_high"], **tally(method, memory)})
    return {"verdict": summary["verdict"], "level": read(RUN / "damage_curve" / "damage_level.json")["fraction"],
            "max_ratio": config["damage_curve"]["max_ratio"], "robots": len(config["robots"]),
            "steps": max(r["steps"] for r in read(RUN / "gate" / "gate.json")),
            "patterns": len(config["final"]["patterns"]), "attempts": episodes,
            "pairs": {"large": config["memory"]["repair_pairs"], "small": config["memory"]["small_repair_pairs"]},
            "conditions": conditions, "cases": sorted(cases, key=lambda c: c["recovery"]),
            "median_case": summary["median_case"], "healthy": tally("healthy", "large"),
            "damaged": tally("none", "large"), "offline": read(RUN / "offline_error.json"),
            "video": video_uri(RUN / "report" / "comparacion.mp4")}


data = read(SITE / "data.json")
data["clips"] = {path.stem: video_uri(path) for path in sorted(SITE.glob("*.mp4"))}
data["official"] = official()
page = (ROOT / "scripts" / "demo_page.html").read_text(encoding="utf-8")
(SITE / "index.html").write_text(page.replace("__DATA__", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
print(SITE / "index.html")
