"""Build dist/robot-rehab.zip with what the other PC needs, and print its path."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["run.sh", "pyproject.toml", "uv.lock", ".python-version", "README.md"]
FOLDERS = ["configs", "src", "tests", "scripts"]


def collect() -> list[Path]:
    paths = [ROOT / name for name in FILES]
    for folder in FOLDERS:
        paths += [p for p in sorted((ROOT / folder).rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts]
    return paths


def main() -> None:
    target = ROOT / "dist" / "robot-rehab.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in collect():
            info = zipfile.ZipInfo.from_file(path, path.relative_to(ROOT).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # Unix: otherwise unzip ignores the permissions when built on Windows
            info.external_attr = (0o100755 if path.name == "run.sh" else 0o100644) << 16
            archive.writestr(info, path.read_bytes())
    print(target)


if __name__ == "__main__":
    main()
