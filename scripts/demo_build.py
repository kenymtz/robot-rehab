"""Put the demo's data and clips and the official results into one page.

Writes runs/demo/site/index.html, and runs/demo/pages with the standalone site for GitHub Pages.

Run after scripts/demo_render.py, the official run and scripts/offline_error.py.
Usage: uv run --frozen python scripts/demo_build.py
"""
import base64
import json
from pathlib import Path

import imageio
import matplotlib
from PIL import Image, ImageDraw, ImageFont

from robotrehab.evaluate import recovery

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "runs" / "demo" / "site"
PAGES = ROOT / "runs" / "demo" / "pages"
RUN = ROOT / "runs" / "full"
URL = "https://kenymtz.github.io/robot-rehab/"
SHARE_FRAME = 120
SUMMARY = "Un robot simulado pierde neuronas e intenta volver a caminar con una memoria de su comportamiento sano."
HEAD = f"""<!doctype html>
<html lang="es">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{SUMMARY}">
<meta property="og:type" content="website">
<meta property="og:url" content="{URL}">
<meta property="og:title" content="¿Puede un robot volver a caminar solo?">
<meta property="og:description" content="{SUMMARY}">
<meta property="og:image" content="{URL}preview.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="627">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' fill='%23425465'/%3E%3Crect x='3' y='3' width='4' height='4' fill='%23edf2ea'/%3E%3Crect x='9' y='3' width='4' height='4' fill='%23edf2ea'/%3E%3Crect x='3' y='9' width='4' height='4' fill='%23edf2ea'/%3E%3Crect x='10' y='10' width='2' height='2' fill='%23d2783d'/%3E%3C/svg%3E">
<style>body {{ margin: 0; }} img {{ max-width: 100%; }}</style>
"""


def share_image(video: Path, out: Path) -> None:
    """Build the picture shown when the link is shared: the headline over one frame of the video."""
    reader = imageio.get_reader(video)
    frame = reader.get_data(SHARE_FRAME)
    reader.close()
    width, height = 1200, 627
    strip = Image.fromarray(frame).resize((width, width * frame.shape[0] // frame.shape[1]))
    image = Image.new("RGB", (width, height), tuple(frame[0, 0]))
    image.paste(strip, (0, height - strip.height))
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"), 52)
    ImageDraw.Draw(image).text((44, 56), "¿Puede un robot volver\na caminar solo?", fill="white", font=font, spacing=12)
    image.save(out)


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
body = page.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
(SITE / "index.html").write_text(body, encoding="utf-8")
print(SITE / "index.html")

PAGES.mkdir(exist_ok=True)
(PAGES / "index.html").write_text(HEAD + body, encoding="utf-8")
(PAGES / ".nojekyll").touch()
share_image(RUN / "report" / "comparacion.mp4", PAGES / "preview.png")
print(PAGES / "index.html")
