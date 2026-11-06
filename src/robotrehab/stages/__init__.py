"""The run, stage by stage. Sources list the code each stage's result depends on."""
from ..pipeline import Stage
from . import damage_curve, final, gate, memory, preflight, report, train, tune

COMMON = ("env_id", "smoke", "robots")

STAGES = [
    Stage("preflight", preflight.run, ("env_id", "robots", "preflight", "train"),
          ("stages/preflight.py",), always=True),
    Stage("train", train.run, COMMON + ("train",), ("stages/train.py",)),
    Stage("gate", gate.run, COMMON + ("train", "gate"),
          ("stages/gate.py", "stages/train.py", "rollout.py", "policy.py")),
    Stage("memory", memory.run, COMMON + ("memory",),
          ("stages/memory.py", "memory.py", "rollout.py", "policy.py")),
    Stage("damage_curve", damage_curve.run, COMMON + ("damage_curve",),
          ("stages/damage_curve.py", "damage.py", "rollout.py")),
    Stage("tune", tune.run, COMMON + ("tune",), ("stages/tune.py", "repair.py", "damage.py")),
    Stage("final", final.run, COMMON + ("final",),
          ("stages/final.py", "repair.py", "damage.py", "rollout.py")),
    Stage("report", report.run, COMMON + ("report",),
          ("stages/report.py", "report.py", "evaluate.py", "video.py")),
]
