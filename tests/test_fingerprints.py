import shutil

from robotrehab.config import load_config
from robotrehab.pipeline import PACKAGE_DIR, PROJECT_ROOT, fingerprint
from robotrehab.stages import STAGES


def test_real_stages_keep_repair_out_of_early_stages(tmp_path):
    copy = tmp_path / "pkg"
    shutil.copytree(PACKAGE_DIR, copy, ignore=shutil.ignore_patterns("__pycache__"))
    cfg = load_config(PROJECT_ROOT / "configs" / "smoke.yaml")

    def keys(package):
        result, upstream = {}, ""
        for stage in STAGES:
            key = fingerprint(stage, cfg, upstream, package)
            if not stage.always:
                upstream = key
            result[stage.name] = key
        return result

    before = keys(copy)
    with open(copy / "repair.py", "a", encoding="utf-8") as handle:
        handle.write("\n# changed\n")
    after = keys(copy)
    for name in ("preflight", "train", "gate", "memory", "damage_curve"):
        assert before[name] == after[name]
    for name in ("tune", "final", "report"):
        assert before[name] != after[name]
