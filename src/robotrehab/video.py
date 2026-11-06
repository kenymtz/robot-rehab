"""Side-by-side video: healthy, damaged, repaired. Runs in its own process so a graphics failure
can never stop the run, and it can be repeated later on any machine from the saved trajectories."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SIZE = 384
STRIDE = 4
FPS = 30
TITLES = ["Sano", "Dañado", "Reparado"]


def label(frame: np.ndarray, text: str) -> np.ndarray:
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    image = Image.fromarray(frame)
    ImageDraw.Draw(image).text((14, 10), text, fill="white", font=ImageFont.truetype(str(font_path), 26))
    return np.array(image)


def compose(panels: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Join panels side by side. A panel that ends early holds its last frame."""
    length = max(len(panel) for panel in panels)
    return [np.hstack([panel[min(i, len(panel) - 1)] for panel in panels]) for i in range(length)]


def render_frames(env_id: str, qpos: np.ndarray, qvel: np.ndarray) -> list[np.ndarray]:
    import gymnasium as gym

    env = gym.make(env_id, render_mode="rgb_array", width=SIZE, height=SIZE)
    env.reset(seed=0)
    frames = []
    for index in range(0, len(qpos), STRIDE):
        env.unwrapped.set_state(qpos[index], qvel[index])
        frames.append(env.render())
    env.close()
    return frames


def main(argv: list[str]) -> int:
    import imageio

    env_id, out, paths = argv[0], argv[1], argv[2:]
    panels = []
    for path, title in zip(paths, TITLES):
        data = np.load(path)
        panels.append([label(f, title) for f in render_frames(env_id, data["qpos"], data["qvel"])])
    imageio.mimsave(out, compose(panels), fps=FPS)
    return 0


def make_video(paths: list[Path], out: Path, env_id: str, log) -> bool:
    """Try each graphics backend in its own process. Returns False if none works."""
    command = [sys.executable, "-m", "robotrehab.video", env_id, str(out)] + [str(p) for p in paths]
    for backend in ("egl", "osmesa", None):
        env = dict(os.environ)
        env.pop("MUJOCO_GL", None)
        if backend:
            env["MUJOCO_GL"] = backend
        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0 and out.exists():
            return True
        log(f"[report] video con {backend or 'modo por defecto'} falló")
    return False


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
