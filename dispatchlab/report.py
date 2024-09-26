"""Launch-decision report: markdown + charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import experiments as E
from .config import Config
from .decision import decide

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"greedy_nearest": "#2a78d6", "ml_batching": "#eb6834", "ortools_assignment": "#1baf7a"}
EST_COLORS = {"ips": "#2a78d6", "dr": "#eb6834"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)


def _fig(*a, **k):
    fig, ax = plt.subplots(*a, **k)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def chart_policies(per_day: pd.DataFrame, path: Path) -> None:
    metrics = [("late_rate", "Late-order rate (%)", 100), ("idle_pct", "Shopper idle time (%)", 100),
               ("total_cost_per_order", "Total cost per order ($)", 1)]
    fig, axes = _fig(1, 3, figsize=(11, 3.8))
    for ax, (m, title, scale) in zip(axes, metrics):
        _style(ax)
        for i, (p, g) in enumerate(per_day.groupby("policy", sort=False)):
            v = g[m].to_numpy() * scale
            ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
            ax.bar(i, v.mean(), width=0.62, color=COLORS[p], zorder=2)
            ax.errorbar(i, v.mean(), yerr=ci, color=INK, capsize=4, linewidth=1.2, zorder=3)
            ax.text(i, v.mean() + ci + 0.01 * max(1, v.mean()), f"{v.mean():.1f}", ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xticks(range(per_day.policy.nunique()))
        ax.set_xticklabels([p.replace("_", "\n") for p in per_day.policy.unique()], fontsize=8, color=INK2)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def chart_by_hour(orders: pd.DataFrame, path: Path) -> None:
    fig, ax = _fig(figsize=(7.5, 3.8))
    _style(ax)
    orders = orders.assign(hour=(8 + orders.arrival_min // 60).astype(int))
    for p, g in orders.groupby("policy", sort=False):
        h = g.groupby("hour").late.mean() * 100
        ax.plot(h.index, h.values, color=COLORS[p], linewidth=2, marker="o", markersize=5, label=p)
    ax.set_xlabel("Hour of order placement", color=INK2)
    ax.set_ylabel("Late-order rate (%)", color=INK2)
    ax.set_title("Lateness concentrates in the lunch and evening peaks", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def chart_calibration(evaluation: dict, path: Path) -> None:
    fig, axes = _fig(1, 2, figsize=(8.5, 3.8))
    for ax, (name, ev) in zip(axes, evaluation.items()):
        _style(ax)
        cal = ev["calibration"]
        lim = [min(cal["pred"] + cal["actual"]) - 1, max(cal["pred"] + cal["actual"]) + 1]
        ax.plot(lim, lim, color=GRID, linewidth=1.5, label="perfect")
        ax.plot(cal["pred"], cal["actual"], color="#2a78d6", linewidth=2, marker="o", markersize=5, label="decile mean")
        ax.set_xlabel("Predicted minutes", color=INK2)
        ax.set_ylabel("Actual minutes", color=INK2)
        ax.set_title(f"{name} time: MAE {ev['mae']:.2f}, p90 error {ev['p90_abs_error']:.2f}", color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def chart_ope(results: pd.DataFrame, path: Path) -> None:
    """Replicate-0 estimates with 95% CIs vs simulator ground truth, lift vs never_batch."""
    r0 = results[(results.replicate == 0) & (results.kind == "lift") & results.estimator.isin(["ips", "dr"])]
    fig, axes = _fig(1, 2, figsize=(10, 3.6))
    for ax, (reward, title, scale) in zip(axes, [("order_cost", "Attributed cost per order: change ($)", 1),
                                                  ("late", "Late-order rate: change (pp)", 100)]):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        pols = list(r0.policy.unique())
        for i, p in enumerate(pols):
            for j, e in enumerate(["ips", "dr"]):
                row = r0[(r0.reward == reward) & (r0.policy == p) & (r0.estimator == e)].iloc[0]
                y = i + (j - 0.5) * 0.25
                ax.errorbar(row.estimate * scale, y, xerr=[[ (row.estimate - row.lo) * scale], [(row.hi - row.estimate) * scale]],
                            fmt="o", color=EST_COLORS[e], capsize=3, markersize=6, label=e.upper() if i == 0 else None)
            t = r0[(r0.reward == reward) & (r0.policy == p)].truth.iloc[0] * scale
            ax.plot(t, i, marker="D", color=INK, markersize=8, linestyle="none", label="Simulator truth" if i == 0 else None)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(range(len(pols)))
        ax.set_yticklabels(pols, color=INK2)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def _pct(x):
    return f"{x * 100:+.1f}%"


def build_report(cfg: Config, evaluation: dict, per_day: pd.DataFrame, orders: pd.DataFrame, ope_out: dict,
                 decision: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_policies(per_day, out_dir / "policy_comparison.png")
    chart_by_hour(orders, out_dir / "late_by_hour.png")
    chart_calibration(evaluation, out_dir / "model_calibration.png")
    chart_ope(ope_out["results"], out_dir / "ope_validation.png")

    summary = per_day.groupby("policy", sort=False).mean(numeric_only=True)
    lifts = decision["lifts"]
    rec = decision["recommended"]
    base = decision["baseline"]
    L = []
    L.append("# DispatchLab launch-decision report\n")
    if decision["launch"]:
        late = lifts[(lifts.policy == rec) & (lifts.metric == "late_rate")].iloc[0]
        cost = lifts[(lifts.policy == rec) & (lifts.metric == "total_cost_per_order")].iloc[0]
        L.append(f"## Recommendation: launch **{rec}**\n")
        L.append(f"Versus `{base}`, expected late-order rate change is **{_pct(late.rel_change)}** "
                 f"(95% CI {_pct(late.lo)} to {_pct(late.hi)}) and total cost per order **{_pct(cost.rel_change)}** "
                 f"(95% CI {_pct(cost.lo)} to {_pct(cost.hi)}), from {cfg.eval_days} paired simulated days.\n")
    else:
        L.append(f"## Recommendation: keep **{base}**\n")
        L.append("No candidate cut the late-order rate with a 95% CI below zero while staying inside the cost guardrail.\n")
    L.append(f"Decision rule: a candidate qualifies if the paired-bootstrap 95% CI of its late-rate change vs `{base}` is "
             f"entirely below zero and total cost per order rises by no more than 2%. Qualifiers: "
             f"{', '.join('`%s`' % q for q in decision['qualifiers']) or 'none'}.\n")

    L.append("## 1. Policy comparison (event-driven simulation)\n")
    L.append(f"{cfg.eval_days} simulated days, identical demand and noise draws for every policy (common random numbers).\n")
    cols = {"late_rate": "late rate", "p90_delivery_min": "p90 delivery (min)", "idle_pct": "shopper idle",
            "attributed_cost_per_order": "attributed $/order", "total_cost_per_order": "total $/order",
            "mean_batch_size": "mean batch size"}
    L.append("| policy | " + " | ".join(cols.values()) + " |")
    L.append("|---|" + "---|" * len(cols))
    for p, r in summary.iterrows():
        cells = [f"{r[c] * 100:.1f}%" if c in ("late_rate", "idle_pct") else f"{r[c]:.2f}" for c in cols]
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L.append("\n![Policy comparison](policy_comparison.png)\n")
    L.append("### Change vs baseline (relative, paired bootstrap 95% CI)\n")
    L.append("| policy | metric | change | 95% CI |\n|---|---|---|---|")
    for r in lifts.itertuples():
        L.append(f"| {r.policy} | {r.metric} | {_pct(r.rel_change)} | {_pct(r.lo)} to {_pct(r.hi)} |")
    L.append("\n![Late rate by hour](late_by_hour.png)\n")

    L.append("## 2. Time-prediction model (held-out days)\n")
    L.append("| target | MAE (min) | baseline MAE | p90 abs error | bias | calibration MAE (deciles) |\n|---|---|---|---|---|---|")
    for n, ev in evaluation.items():
        L.append(f"| {n} | {ev['mae']:.2f} | {ev['baseline_mae']:.2f} | {ev['p90_abs_error']:.2f} | {ev['bias']:+.3f} | {ev['calibration_mae']:.3f} |")
    L.append("\n![Calibration](model_calibration.png)\n")

    res = ope_out["results"]
    L.append("## 3. Off-policy evaluation vs simulator ground truth\n")
    L.append(f"Logs come from a cluster-randomized policy (action drawn per store and 30-minute slot, 50% exploration, "
             f"known propensities). {cfg.ope_replicates} independent logged datasets x {cfg.ope_log_days} days were evaluated; "
             f"ground truth is each target policy simulated online for {cfg.ope_truth_days} fresh days. "
             f"CIs are day-level bootstraps. Targets run under nearest-shopper assignment, so they isolate the batching decision.\n")
    L.append("**Ground truth (simulator):**\n")
    L.append("| policy | late rate | attributed $/order |\n|---|---|---|")
    tr = ope_out["truth"].pivot(index="policy", columns="reward", values="truth")
    for p, r in tr.iterrows():
        L.append(f"| {p} | {r['late'] * 100:.1f}% | {r['order_cost']:.2f} |")
    L.append("\n**Replicate-0 estimates of the change vs `never_batch` (95% CI):**\n")
    r0 = res[(res.replicate == 0) & (res.kind == "lift") & res.estimator.isin(["ips", "snips", "dr"])]
    L.append("| policy | metric | truth | IPS | SNIPS | DR |\n|---|---|---|---|---|---|")
    for (p, rw), g in r0.groupby(["policy", "reward"], sort=False):
        sc = 100 if rw == "late" else 1
        unit = "pp" if rw == "late" else "$"
        cell = lambda e: (lambda x: f"{x.estimate * sc:+.2f} [{x.lo * sc:+.2f}, {x.hi * sc:+.2f}]")(g[g.estimator == e].iloc[0])
        L.append(f"| {p} | {rw} ({unit}) | {g.truth.iloc[0] * sc:+.2f} | {cell('ips')} | {cell('snips')} | {cell('dr')} |")
    L.append("\n![OPE vs truth](ope_validation.png)\n")
    L.append(f"**CI coverage and error across all {cfg.ope_replicates} replicates** (nominal coverage 95%):\n")
    cov = res[res.estimator != "dm"].groupby(["reward", "kind", "estimator"]).agg(
        coverage=("covered", "mean"), mean_abs_error=("abs_err", "mean")).reset_index()
    L.append("| reward | kind | estimator | CI coverage | mean abs error |\n|---|---|---|---|---|")
    for r in cov.itertuples():
        L.append(f"| {r.reward} | {r.kind} | {r.estimator} | {r.coverage * 100:.0f}% | {r.mean_abs_error:.3f} |")
    ess = ope_out["ess"].groupby("policy").ess_fraction.mean()
    L.append("\nEffective sample size as a fraction of logged orders: " + ", ".join(f"{p} {v:.0%}" for p, v in ess.items()) + ".\n")
    late_cov = cov[(cov.reward == "late") & (cov.kind == "lift") & (cov.estimator == "dr")].coverage.iloc[0]
    cost_cov = cov[(cov.reward == "order_cost") & (cov.kind == "lift") & (cov.estimator == "dr")].coverage.iloc[0]
    L.append("### How to read this\n")
    L.append(f"- Per-order **cost** is a direct effect of the batching action, and OPE recovers its sign and rough size (DR lift CI coverage {cost_cov:.0%}; nominal 95%, so treat the CIs as optimistic).")
    L.append(f"- **Lateness** also depends on a congestion externality: batching in one slot frees shoppers for later orders. "
             f"Logs where only some slots batch cannot see that spillover, so OPE's late-rate lift can be biased (DR lift CI coverage {late_cov:.0%}). "
             "This is exactly why the launch decision above is made on the simulator, with OPE used as a fast pre-screen "
             "for direct effects.\n")

    L.append("## 4. Caveats\n")
    L.append("- All data is synthetic; absolute numbers illustrate the method, not a real market.")
    L.append("- `idle` is paid shift time not spent travelling, picking or delivering; it is not a per-order OPE target.")
    L.append("- The simulator is a model: confirm the winner with a small live holdout before a full launch.\n")
    path = out_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def run_all(cfg: Config, log=print) -> Path:
    log(f"[1/5] Generating synthetic data (Parquet) and training the pick/delivery time model -> {cfg.data_dir}/")
    world, model, evaluation = E.prepare(cfg)
    log(f"      held-out MAE: pick {evaluation['pick']['mae']:.2f} min, drive {evaluation['drive']['mae']:.2f} min")
    log(f"[2/5] Simulating 3 policies x {cfg.eval_days} days (common random numbers)")
    per_day, orders = E.compare_policies(world, cfg, model)
    decision = decide(per_day, cfg.bootstrap)
    log(f"      decision: {decision['recommended']}")
    log(f"[3/5] Off-policy evaluation: {cfg.ope_replicates} logged datasets, ground truth from {cfg.ope_truth_days} days per target")
    ope_out = E.ope_validation(world, cfg, model)
    log("[4/5] Writing report and charts")
    out = build_report(cfg, evaluation, per_day, orders, ope_out, decision, cfg.output_dir)
    per_day.to_csv(cfg.output_dir / "policy_metrics_by_day.csv", index=False)
    ope_out["results"].to_csv(cfg.output_dir / "ope_results.csv", index=False)
    log("[5/5] Done")
    return out
