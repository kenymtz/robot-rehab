"""Turn the final records into a verdict, a report and a figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate import bootstrap_median_ci, diagnosis, recovery, verdict

MAIN = "mutation/large"
LABELS = {
    "mutation/large": "Mutación con memoria (principal)",
    "ridge/large": "Mínimos cuadrados",
    "asr/large": "Reescalado de actividad",
    "bc/large": "Clonación con gradientes",
    "mutation/small": "Mutación, memoria chica",
    "ridge/small": "Mínimos cuadrados, memoria chica",
    "bc/small": "Clonación, memoria chica",
}
VERDICTS = {
    "seguir": "**Seguir** a la fase 2.",
    "investigar": "**Investigar** qué casos fallan.",
    "explicar": "**Explicar**: el proyecto pasa a analizar por qué falla.",
    "sin_datos": "**Sin datos**: ningún caso quedó degradado.",
}
DIAGNOSES = {
    "busqueda": "La clonación recupera y la mutación no: el problema es la búsqueda.",
    "memoria_o_bucle": "La clonación tampoco recupera: el límite está en la memoria o en el bucle cerrado.",
    "ridge_principal": "Mínimos cuadrados supera a la mutación: el método simple pasa a ser el principal.",
}


def _case_recoveries(records: list[dict]) -> tuple[dict, list]:
    healthy = {r["robot"]: r["return"] for r in records if r["condition"] == "healthy"}
    damaged = {(r["robot"], r["pattern"]): r["return"] for r in records if r["condition"] == "none"}
    excluded = [{"robot": robot, "pattern": pattern} for (robot, pattern), value in damaged.items()
                if recovery(value, value, healthy[robot]) is None]
    values: dict[str, list[dict]] = {}
    for r in records:
        if r["condition"] in ("healthy", "none"):
            continue
        case = (r["robot"], r["pattern"])
        result = recovery(r["return"], damaged[case], healthy[r["robot"]])
        if result is not None:
            values.setdefault(f"{r['condition']}/{r['memory']}", []).append(
                {"robot": r["robot"], "pattern": r["pattern"], "recovery": result})
    return values, excluded


def _describe(cases: list[dict], n_resamples: int) -> dict:
    median, low, high = bootstrap_median_ci([c["recovery"] for c in cases], n_resamples, seed=0)
    robots: dict[int, list[float]] = {}
    for case in cases:
        robots.setdefault(case["robot"], []).append(case["recovery"])
    by_robot = {str(robot): sorted(v)[len(v) // 2] for robot, v in robots.items()}
    return {"median": median, "ci_low": low, "ci_high": high, "n_cases": len(cases), "by_robot": by_robot}


def build_summary(records: list[dict], n_resamples: int) -> dict:
    values, excluded = _case_recoveries(records)
    conditions = {key: _describe(cases, n_resamples) for key, cases in values.items()}
    warnings = []
    if len(excluded) > 5:
        warnings.append(f"Se excluyeron {len(excluded)} casos sin degradación, más de 5.")
    main = conditions.get(MAIN)
    medians = {c["name"]: conditions[c["key"]]["median"] for c in
               ({"name": "mutation", "key": MAIN}, {"name": "ridge", "key": "ridge/large"},
                {"name": "bc", "key": "bc/large"}) if c["key"] in conditions}
    median_case = None
    if main is not None:
        ordered = sorted(values[MAIN], key=lambda c: c["recovery"])
        median_case = ordered[len(ordered) // 2]
    return {
        "conditions": conditions, "excluded": excluded, "warnings": warnings,
        "verdict": verdict(main["median"], main["ci_low"]) if main else "sin_datos",
        "diagnosis": diagnosis(medians) if len(medians) == 3 else [],
        "median_case": median_case,
    }


def render_markdown(summary: dict, meta: dict) -> str:
    lines = ["# Resultados de robot-rehab", ""]
    if meta["smoke"]:
        lines += ["> **SIN VALIDEZ CIENTÍFICA.** Prueba de la tubería con robots casi sin entrenar.", ""]
    lines += [f"**Veredicto:** {VERDICTS[summary['verdict']]}", ""]
    lines += [f"- {DIAGNOSES[code]}" for code in summary["diagnosis"]]
    level = meta["level"]
    note = " Ningún nivel alcanzó la caída buscada: se usó el valor por defecto." if level["used_fallback"] else ""
    lines += ["", f"**Nivel de daño:** {level['fraction']:.0%} de las neuronas.{note}", "",
              "| Condición | Mediana de recuperación | IC 95% | Casos | Por robot |", "|---|---|---|---|---|"]
    for key, label in LABELS.items():
        if key in summary["conditions"]:
            c = summary["conditions"][key]
            robots = ", ".join(f"{r}: {v:.0%}" for r, v in c["by_robot"].items())
            lines.append(f"| {label} | {c['median']:.0%} | {c['ci_low']:.0%} a {c['ci_high']:.0%} "
                         f"| {c['n_cases']} | {robots} |")
    lines += ["", "![Recuperación por condición](recuperacion.png)", ""]
    if summary["excluded"]:
        cases = ", ".join(f"robot {c['robot']} patrón {c['pattern']}" for c in summary["excluded"])
        lines += [f"**Casos excluidos por no degradarse:** {cases}.", ""]
    lines += [f"**Advertencia:** {warning}" for warning in summary["warnings"]]
    if not meta["video"]:
        lines += ["", "**Video:** no se pudo generar aquí. Las trayectorias están en `traj/`."]
    return "\n".join(lines) + "\n"


def plot_recovery(summary: dict, path) -> None:
    """Median and 95% interval per condition, against the two decision thresholds."""
    keys = [key for key in LABELS if key in summary["conditions"]][::-1]
    figure, axes = plt.subplots(figsize=(8, 0.55 * max(len(keys), 1) + 1.4), dpi=150)
    for row, key in enumerate(keys):
        c = summary["conditions"][key]
        axes.plot([c["ci_low"], c["ci_high"]], [row, row], color="#2a6fbb", linewidth=2,
                  solid_capstyle="round")
        axes.plot(c["median"], row, "o", color="#2a6fbb", markersize=8,
                  markeredgecolor="white", markeredgewidth=2)
    for threshold in (0.1, 0.5):
        axes.axvline(threshold, color="#9aa3ad", linewidth=1, linestyle=(0, (4, 3)), zorder=0)
    axes.axvline(0, color="#5c6670", linewidth=1, zorder=0)
    axes.set_yticks(range(len(keys)), [LABELS[key] for key in keys])
    axes.set_xlabel("Rendimiento perdido que volvió (1 = recuperación total)")
    axes.set_title("Recuperación por condición: mediana e intervalo del 95%", loc="left", fontsize=11)
    axes.spines[["top", "right", "left"]].set_visible(False)
    axes.tick_params(axis="y", length=0)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
