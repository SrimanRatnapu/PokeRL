"""Make every figure in docs/figures/ from the training logs and the evaluation results.

Two steps:
  1. export: TensorBoard event files in logs/<run>/ -> results/curves/<run>.csv
     (runs with several event files, i.e. resumed runs, are stitched by step).
     Skipped for runs whose logs/<run>/ folder is missing, so the figures can be
     rebuilt from the committed CSVs alone.
  2. plot: results/curves/*.csv, results/evals.csv and logs/e10-beta-sweep.log
     -> docs/figures/*.png

    .venv/bin/python scripts/make_figures.py            # export (if logs exist) + plot
    .venv/bin/python scripts/make_figures.py --no-export
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
CURVES = ROOT / "results" / "curves"
EVALS = ROOT / "results" / "evals.csv"
FIGS = ROOT / "docs" / "figures"

# run name -> label. Order = order in the plots.
RUNS = {
    "v1": "v1 features, from scratch (E1-E2)",
    "v2bc": "v2 + behaviour-cloning warm start (E4)",
    "v2scratch": "v2 from scratch = champion (E5-E6)",
    "v2ft": "v2ft: champion +2M (E7 control)",
    "v3ft": "v3ft: + items/crits/history (E7)",
    "v3ft2": "v3ft2 (E8 control)",
    "v4ft": "v4ft: + abilities (E8)",
    "league": "league: PFSP + strong pool + exploiter (E9)",
    "exploiter": "exploiter (E9)",
    "v5ft": "v5ft: + lookahead scores (E10)",
}
TAGS = [
    "curriculum/stage", "curriculum/pool_size", "reward/shaping_scale",
    "rollout/ep_rew_mean", "rollout/ep_len_mean", "time/fps",
    "winrate/rolling_all", "winrate/random", "winrate/maxpower", "winrate/heuristic",
    "winrate/simple_heuristic", "winrate/policy", "winrate/lookahead",
    "train/approx_kl", "train/clip_fraction", "train/entropy_loss", "train/explained_variance",
    "train/learning_rate", "train/loss", "train/policy_gradient_loss", "train/value_loss",
    "eval/win_rate_vs_simple_heuristic",
]
STAGE_NAMES = {1: "1: random", 2: "2: max-power", 3: "3: heuristics", 4: "4: self-play"}
STAGE_COLORS = {1: "#f2f2f2", 2: "#e6eef7", 3: "#fbeedd", 4: "#e8f4e8"}
OPP_COLORS = {
    "random": "#9e9e9e", "maxpower": "#7b61a8", "heuristic": "#d9822b", "simple_heuristic": "#c0392b",
    "policy": "#2e7d32", "lookahead": "#1565c0",
}
OPP_LABELS = {
    "random": "random", "maxpower": "max-power", "heuristic": "HeuristicBot",
    "simple_heuristic": "SimpleHeuristics", "policy": "self-play pool", "lookahead": "lookahead bot",
}
RUN_COLORS = {
    "v1": "#9e9e9e", "v2bc": "#7b61a8", "v2scratch": "#000000", "v2ft": "#8d6e63", "v3ft": "#d9822b",
    "v3ft2": "#f2b880", "v4ft": "#1565c0", "league": "#2e7d32", "exploiter": "#c0392b", "v5ft": "#c2185b",
}


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_curves() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    CURVES.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        files = sorted(glob.glob(str(LOGS / run / "**" / "events.out.tfevents.*"), recursive=True))
        if not files:
            print(f"[export] {run}: no TensorBoard logs, keeping {run}.csv as is")
            continue
        frames = []
        for f in files:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
            cols = {}
            for tag in TAGS:
                if tag in ea.Tags()["scalars"]:
                    cols[tag] = pd.Series({e.step: e.value for e in ea.Scalars(tag)})
            if cols:
                frames.append(pd.DataFrame(cols))
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]  # a resumed run re-logs its first step
        df.index.name = "step"
        df.to_csv(CURVES / f"{run}.csv", float_format="%.6g")
        print(f"[export] {run}: {len(files)} event file(s), {len(df)} rows, steps {df.index.min():,}-{df.index.max():,}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load(run: str) -> pd.DataFrame:
    return pd.read_csv(CURVES / f"{run}.csv", index_col="step")


def smooth(s: pd.Series, k: int = 9) -> pd.Series:
    s = s.dropna()
    return s.rolling(k, center=True, min_periods=1).mean()


def wilson(w: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def millions(ax) -> None:
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))


def pct(ax, axis: str = "y") -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def stage_bands(ax, df: pd.DataFrame, label: bool = True) -> None:
    st = df["curriculum/stage"].dropna()
    if st.empty:
        return
    steps, vals = st.index.to_numpy(), st.to_numpy().astype(int)
    start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[start]:
            x0 = steps[start] if start else 0
            x1 = steps[i] if i < len(vals) else steps[-1]
            ax.axvspan(x0, x1, color=STAGE_COLORS[vals[start]], zorder=0, lw=0)
            if label and (x1 - x0) > 0.04 * steps[-1]:
                ax.text((x0 + x1) / 2, 1.0, STAGE_NAMES[vals[start]], ha="center", va="top",
                        fontsize=7.5, color="#555", transform=ax.get_xaxis_transform())
            start = i


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] {path.relative_to(ROOT)}")


def evals() -> pd.DataFrame:
    df = pd.read_csv(EVALS)
    df["rate"] = df.wins / df.battles
    ci = [wilson(w, n) for w, n in zip(df.wins, df.battles)]
    df["lo"] = [c[0] for c in ci]
    df["hi"] = [c[1] for c in ci]
    return df


def plot_training_winrates(ax, df: pd.DataFrame, opps=None, k: int = 9) -> None:
    for opp in opps or OPP_COLORS:
        col = f"winrate/{opp}"
        if col in df and df[col].notna().sum() > 3:
            raw = df[col].dropna()
            changed = raw.ne(raw.shift())
            raw = raw.loc[: changed[changed].index[-1]]  # drop the frozen tail after the last game vs it
            s = smooth(raw, k)
            ax.plot(s.index, s.values, color=OPP_COLORS[opp], lw=1.4, label=OPP_LABELS[opp])


def plot_eval(ax, df: pd.DataFrame, color: str = "black", label: str | None = "200-battle eval vs SimpleHeuristics",
              band: bool = True, raw: bool = True, k: int = 5) -> None:
    e = df["eval/win_rate_vs_simple_heuristic"].dropna()
    if e.empty:
        return
    if band:
        lo, hi = zip(*[wilson(p * 200, 200) for p in e.values])
        ax.fill_between(e.index, lo, hi, color=color, alpha=0.10, lw=0)
    if raw:
        ax.plot(e.index, e.values, "o", ms=2.5, color=color, alpha=0.45)
    s = smooth(e, k)
    ax.plot(s.index, s.values, color=color, lw=2, label=label)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_champion_curriculum() -> None:
    df = load("v2scratch")
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    stage_bands(a, df)
    plot_training_winrates(a, df, ["random", "maxpower", "heuristic", "simple_heuristic", "policy"])
    a.set_ylabel("training win rate\n(last 200 games vs that opponent)")
    a.set_ylim(0, 1.05)
    pct(a)
    a.axvline(1_000_000, color="#555", ls=":", lw=1)
    a.text(1_010_000, 0.04, "resumed for +3M (D18)", fontsize=8, color="#555")
    h, _ = a.get_legend_handles_labels()
    h += [matplotlib.patches.Patch(color=STAGE_COLORS[i], label=f"stage {STAGE_NAMES[i]}") for i in STAGE_COLORS]
    a.legend(handles=h, loc="lower right", fontsize=8, ncol=3, framealpha=0.9)
    a.set_title("Champion run (v2 features, PPO from scratch, 4M steps): win rate through the curriculum", fontsize=11)

    stage_bands(b, df, label=False)
    plot_eval(b, df)
    b.axhline(0.5, color="#888", lw=0.8)
    b.set_ylabel("eval win rate\nvs SimpleHeuristics")
    b.set_ylim(0.2, 0.9)
    pct(b)
    millions(b)
    b.set_xlabel("environment steps")
    b.legend(loc="lower right", fontsize=8)
    save(fig, "01_champion_curriculum.png")


def fig_ppo_diagnostics(run: str = "v2scratch", name: str = "02_champion_ppo_diagnostics.png") -> None:
    df = load(run)
    panels = [
        ("rollout/ep_rew_mean", "mean episode return (win ±30 + shaping)"),
        ("rollout/ep_len_mean", "mean episode length (decisions)"),
        ("train/entropy_loss", "entropy loss (−entropy)"),
        ("train/value_loss", "value loss"),
        ("train/policy_gradient_loss", "policy-gradient loss"),
        ("train/explained_variance", "explained variance of the critic"),
        ("train/approx_kl", "approx. KL per update"),
        ("train/clip_fraction", "clip fraction"),
        ("train/learning_rate", "learning rate / shaping scale"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        stage_bands(ax, df, label=False)
        if col in df:
            s = df[col].dropna()
            ax.plot(s.index, s.values, color="#90a4ae", lw=0.6)
            ss = smooth(s, 15)
            ax.plot(ss.index, ss.values, color="#263238", lw=1.5)
        if col == "train/learning_rate":
            ax.set_yscale("log")
            ax2 = ax.twinx()
            sh = df["reward/shaping_scale"].dropna()
            ax2.plot(sh.index, sh.values, color="#d9822b", lw=1.5)
            ax2.set_ylabel("shaping scale", color="#d9822b", fontsize=8)
            ax2.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        millions(ax)
    for ax in axes[-1]:
        ax.set_xlabel("environment steps", fontsize=8)
    present = sorted(set(df["curriculum/stage"].dropna().astype(int)))
    handles = [matplotlib.patches.Patch(color=STAGE_COLORS[i], label=f"stage {STAGE_NAMES[i]}") for i in present]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"PPO training diagnostics: {RUNS[run]}", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, name)


def fig_from_scratch_runs() -> None:
    """E1-E5: v1 vs v2 features vs BC warm start: curriculum speed and eval curves."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    for run in ["v1", "v2scratch", "v2bc"]:
        df = load(run)
        st = df["curriculum/stage"].dropna()
        a.step(st.index, st.values, where="post", color=RUN_COLORS[run], lw=2, label=RUNS[run])
        plot_eval(b, df, color=RUN_COLORS[run], label=RUNS[run], band=False, raw=False, k=7)
    a.set_yticks([1, 2, 3, 4], [STAGE_NAMES[i] for i in range(1, 5)])
    a.set_xlim(0, 1.1e6)
    a.set_title("Curriculum: when each run reached each stage", fontsize=10)
    a.legend(fontsize=8, loc="lower right")
    millions(a)
    a.set_xlabel("environment steps")
    b.axhline(0.5, color="#888", lw=0.8)
    b.set_title("200-battle eval vs SimpleHeuristics (smoothed)", fontsize=10)
    b.set_ylim(0.2, 0.85)
    pct(b)
    millions(b)
    b.set_xlabel("environment steps")
    b.legend(fontsize=8, loc="lower right")
    fig.suptitle("E1-E5: better features (v2) beat both more steps (v1) and imitation (BC)", fontsize=11)
    fig.tight_layout()
    save(fig, "03_from_scratch_runs.png")


def fig_finetune_runs() -> None:
    """E7-E10: every 2M-step fine-tune from the champion's weights."""
    runs = ["v2ft", "v3ft", "v3ft2", "v4ft", "league", "v5ft"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for run in runs:
        df = load(run)
        c = RUN_COLORS[run]
        plot_eval(axes[0], df, color=c, label=RUNS[run], band=False, raw=False, k=9)
        for ax, opp in zip(axes[1:], ["heuristic", "simple_heuristic"]):
            s = smooth(df[f"winrate/{opp}"], 25)
            ax.plot(s.index, s.values, color=c, lw=1.5, label=RUNS[run])
    titles = ["200-battle eval vs SimpleHeuristics (smoothed)",
              "training win rate vs HeuristicBot (last 200 games)",
              "training win rate vs SimpleHeuristics (last 200 games)"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)
        ax.set_ylim(0.5, 0.95)
        pct(ax)
        millions(ax)
        ax.set_xlabel("environment steps")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("E7-E10: all fine-tunes start near the champion; only v5ft ends measurably above its matched control",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "04_finetune_runs.png")


def fig_league() -> None:
    df = load("league")
    ex = load("exploiter")
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    plot_training_winrates(a, df, ["policy", "lookahead", "heuristic", "simple_heuristic", "maxpower"], k=15)
    a.axvline(1_000_000, color="#555", ls=":", lw=1)
    a.text(1_015_000, 0.975, "phase B: exploiter joins the pool", fontsize=8, color="#555", va="top")
    a.axhline(0.5, color="#888", lw=0.8)
    a.set_ylim(0.3, 1.0)
    pct(a)
    millions(a)
    a.set_xlabel("environment steps")
    a.set_ylabel("training win rate (last 200 games)")
    a.set_title("League run: win rate by opponent type", fontsize=10)
    a.legend(fontsize=8, loc="lower right", ncol=2)
    s = smooth(ex["winrate/policy"], 9)
    b.plot(s.index, s.values, color=OPP_COLORS["policy"], lw=1.6, label="vs frozen main agent (85% of games)")
    s = smooth(ex["winrate/simple_heuristic"], 9)
    b.plot(s.index, s.values, color=OPP_COLORS["simple_heuristic"], lw=1.6, label="vs SimpleHeuristics (15%)")
    b.axhline(0.5, color="#888", lw=0.8)
    b.set_ylim(0.3, 1.0)
    pct(b)
    millions(b)
    b.set_xlabel("environment steps")
    b.set_title("Exploiter: barely beats its target", fontsize=10)
    b.legend(fontsize=8, loc="lower right")
    fig.suptitle("E9: league training. PFSP had little to prioritise: no opponent was clearly beating the agent",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "05_league_training.png")


def fig_progress() -> None:
    """Every milestone model vs the scripted bots, 300-2000 battles each, with 95% CIs."""
    ev = evals()
    sc = ev[ev.kind == "scripted"]
    order = ["v1 @1M", "v1 @3M", "BC clone", "v2 + BC @2M", "v2 scratch @1M", "champion (v2 @4M)",
             "v2ft", "v3ft", "v3ft2", "v4ft", "league", "v5ft"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(order))
    for j, (opp, off) in enumerate([("maxpower", -0.2), ("heuristic", 0.0), ("simple_heuristic", 0.2)]):
        rows = []
        for m in order:
            r = sc[(sc.model == m) & (sc.opponent == opp)]
            rows.append(r.sort_values("battles").iloc[-1] if len(r) else None)  # the largest eval
        xs = [x[i] + off for i, r in enumerate(rows) if r is not None]
        ys = [r.rate for r in rows if r is not None]
        err = [[r.rate - r.lo for r in rows if r is not None], [r.hi - r.rate for r in rows if r is not None]]
        ax.errorbar(xs, ys, yerr=err, fmt="o", ms=5, capsize=2.5, color=OPP_COLORS[opp], label=OPP_LABELS[opp])
    ax.axvspan(5 - 0.45, 5 + 0.45, color="#fff3c4", zorder=0)
    ax.set_xticks(x, [m.replace(" (v2 @4M)", "\n(v2 @4M)").replace(" @", "\n@") for m in order], fontsize=8.5)
    ax.set_ylim(0.35, 1.0)
    pct(ax)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("win rate (95% Wilson CI)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Every milestone model vs the scripted bots (300-2000 battles each; largest eval shown)", fontsize=11)
    save(fig, "06_progress_vs_scripted_bots.png")


def _forest(ax, rows: pd.DataFrame, labels: list[str], colors: list[str]) -> None:
    y = np.arange(len(rows))[::-1]
    for yi, (_, r), c in zip(y, rows.iterrows(), colors):
        ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=2.2)
        ax.plot(r.rate, yi, "o", color=c, ms=6)
        ax.text(r.hi + 0.005, yi, f"{r.rate:.1%}  (n={r.battles})", va="center", fontsize=8)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.axvline(0.5, color="#555", lw=1)
    pct(ax, "x")
    ax.grid(axis="x", alpha=0.3)


def fig_head_to_heads() -> None:
    ev = evals()
    h = ev[ev.kind == "h2h"].reset_index(drop=True)
    labels = [f"{r.experiment}: {r.model}  vs  {r.opponent}" for _, r in h.iterrows()]
    colors = ["#2e7d32" if r.lo > 0.5 else ("#c0392b" if r.hi < 0.5 else "#757575") for _, r in h.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 0.33 * len(h) + 1.2))
    _forest(ax, h, labels, colors)
    ax.set_xlim(0.38, 0.75)
    ax.set_xlabel("win rate of the first model (95% Wilson CI); green / red = significantly above / below 50%, grey = no significant difference")
    ax.set_title("Every head-to-head match in the project", fontsize=11)
    save(fig, "07_head_to_heads.png")


def beta_sweep() -> pd.DataFrame:
    rows = []
    pat = re.compile(r"^beta=([\d.]+) vs (lookahead|champion):.*?(\d+)/(\d+)\s")
    for line in (ROOT / "logs" / "e10-beta-sweep.log").read_text().splitlines():
        m = pat.match(line)
        if m:
            rows.append(dict(beta=float(m[1]), opp=m[2], wins=int(m[3]), battles=int(m[4])))
    df = pd.DataFrame(rows)
    df["rate"] = df.wins / df.battles
    ci = [wilson(w, n) for w, n in zip(df.wins, df.battles)]
    df["lo"], df["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    df.to_csv(ROOT / "results" / "beta_sweep.csv", index=False, float_format="%.4f")
    return df


def fig_beta_sweep() -> None:
    df = beta_sweep()
    ev = evals()
    look_champ = ev[(ev.model == "lookahead bot") & (ev.opponent == "champion (v2 @4M)")].iloc[0]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for opp, c, lab in [("champion", "#000000", "vs champion"), ("lookahead", OPP_COLORS["lookahead"], "vs lookahead bot")]:
        d = df[df.opp == opp].sort_values("beta")
        x = d.beta.replace(0, 0.1)  # log axis: draw beta = 0 at 0.1
        ax.fill_between(x, d.lo, d.hi, color=c, alpha=0.12, lw=0)
        ax.plot(x, d.rate, "o-", color=c, lw=1.8, label=f"v4ft + search {lab}")
    ax.axhline(look_champ.rate, color="#000", ls="--", lw=1)
    ax.text(0.11, look_champ.rate + 0.006, f"lookahead bot alone vs champion: {look_champ.rate:.1%}", fontsize=8)
    ax.axhline(0.5, color="#888", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.5, 1, 2, 4, 8, 100], ["0\n(plain policy)", "0.5", "1", "2", "4", "8", "100\n(lookahead decides)"])
    ax.minorticks_off()
    pct(ax)
    ax.set_xlabel("β in  argmax  log π(a) + β · lookahead(a)")
    ax.set_ylabel("win rate (2000 battles, 95% CI)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("E10 β sweep: more search weight is better; the policy is a good tie-breaker", fontsize=11)
    save(fig, "08_search_beta_sweep.png")


def fig_lookahead_ablation() -> None:
    ev = evals()
    ab = ev[ev.kind == "ablation"].reset_index(drop=True)
    labels = [m.replace("lookahead: ", "") for m in ab.model]
    colors = ["#1565c0"] * 3 + ["#90a4ae"] * (len(ab) - 3)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    _forest(ax, ab, labels, colors)
    ax.set_xlim(0.45, 0.66)
    ax.set_xticks(np.arange(0.45, 0.651, 0.05))
    ax.set_xlabel("win rate vs the E9 lookahead bot (2000 battles, 95% CI)")
    ax.set_title("D23 lookahead ablation: nearly all of the gain is terastallization (grey = tuning, all within noise)",
                 fontsize=10)
    save(fig, "09_lookahead_ablation.png")


def fig_headline() -> None:
    ev = evals()
    want = [("v4ft", "plain policy (v4ft)"), ("v5ft", "plain policy, v5 features (v5ft)"),
            ("lookahead bot", "lookahead bot alone (no learning)"),
            ("v5ft + search", "v5ft + lookahead search (β = 100)"),
            ("v4ft + search", "v4ft + lookahead search (β = 100)")]
    rows = pd.concat([ev[(ev.model == m) & (ev.opponent == "champion (v2 @4M)") & (ev.battles == 2000)] for m, _ in want])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    _forest(ax, rows.reset_index(drop=True), [lab for _, lab in want],
            ["#1565c0", "#c2185b", "#757575", "#c2185b", "#1565c0"])
    ax.set_xlim(0.45, 0.72)
    ax.set_xlabel("win rate vs the champion (2000 battles, 95% Wilson CI)")
    ax.set_title("Headline (E10): search at decision time adds 8-14 points against the champion", fontsize=11)
    save(fig, "00_headline_vs_champion.png")


def fig_all_runs_overview() -> None:
    """Eval curve of every run on one axis, by total training steps behind the model.

    The E7-E10 fine-tunes all start from the champion's weights (4M steps), so they are drawn from 4M on.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for run in ["v1", "v2bc", "v2scratch", "v2ft", "v3ft", "v3ft2", "v4ft", "league", "v5ft"]:
        df = load(run)
        if run not in ("v1", "v2bc", "v2scratch"):
            df = df.set_axis(df.index + 4_000_000, axis=0)
        plot_eval(ax, df, color=RUN_COLORS[run], label=RUNS[run], band=False, raw=False, k=9)
    ax.axhline(0.5, color="#888", lw=0.8)
    ax.set_ylim(0.2, 0.9)
    pct(ax)
    millions(ax)
    ax.axvline(4_000_000, color="#555", ls=":", lw=1)
    ax.text(4_030_000, 0.88, "E7-E10 fine-tunes start from the champion", fontsize=8, color="#555")
    ax.set_xlabel("total environment steps behind the model")
    ax.set_ylabel("200-battle eval\nvs SimpleHeuristics")
    ax.legend(fontsize=7.5, ncol=2, loc="lower left", bbox_to_anchor=(0.2, 0.0))
    ax.set_title("Eval learning curves of every training run (smoothed)", fontsize=11)
    save(fig, "10_all_runs_eval_curves.png")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-export", action="store_true", help="plot from results/curves/*.csv only")
    args = p.parse_args()
    if not args.no_export:
        export_curves()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig_headline()
    fig_champion_curriculum()
    fig_ppo_diagnostics()
    fig_from_scratch_runs()
    fig_finetune_runs()
    fig_league()
    fig_progress()
    fig_head_to_heads()
    fig_beta_sweep()
    fig_lookahead_ablation()
    fig_all_runs_overview()
    fig_ppo_diagnostics("v5ft", "11_v5ft_ppo_diagnostics.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
