"""Put the demo's real data and clips into one page: runs/demo/site/index.html."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "runs" / "demo" / "site"

data = json.loads((SITE / "data.json").read_text(encoding="utf-8"))
data["clips"] = {path.stem: "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()
                 for path in sorted(SITE.glob("*.mp4"))}
page = (ROOT / "scripts" / "demo_page.html").read_text(encoding="utf-8")
(SITE / "index.html").write_text(page.replace("__DATA__", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
print(SITE / "index.html")
