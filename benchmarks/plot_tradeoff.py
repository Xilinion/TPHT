#!/usr/bin/env python3
"""Space-efficiency vs throughput tradeoff, in the style of the paper's figure.

Reads results/tpht_space_eff.csv plus one results/tpht_latency CSV per load
factor and pairs them: for each (variant, key width) the space efficiency at a
load factor comes from the space sweep and the throughput from the latency
sweep at that same load factor.

Usage:
    plot_tradeoff.py space_eff.csv out.pdf LF=latency.csv [LF=latency.csv ...]
e.g.
    plot_tradeoff.py results/tpht_space_eff.csv results/tpht_tradeoff.pdf \
        0.50=lat_lf0.50.csv 0.70=lat_lf0.70.csv 0.85=lat_lf0.85.csv
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    space_csv, out = sys.argv[1], sys.argv[2]
    lat_by_lf = {}
    for arg in sys.argv[3:]:
        lf, path = arg.split("=", 1)
        lat_by_lf[float(lf)] = path

    # space efficiency: (variant, key_bits, target_load) -> % (value_bits=64
    # to match the latency bench's 8-byte values)
    space = {}
    for r in csv.DictReader(open(space_csv)):
        if r["value_bits"] != "64":
            continue
        key = (r["variant"], r["key_bits"], round(float(r["target_load"]), 2))
        space[key] = float(r["space_efficiency"]) * 100.0

    # throughput: (variant, key_bits, lf, phase) -> Mops/s
    thpt = {}
    for lf, path in lat_by_lf.items():
        for r in csv.DictReader(open(path)):
            key = (r["variant"], r["key_bits"], lf, r["phase"])
            thpt[key] = float(r["ops_per_sec"]) / 1e6

    series = [
        ("chained-tpht", "32", "tab:blue", "o"),
        ("chained-tpht", "64", "tab:cyan", "s"),
        ("flatten-tpht", "32", "tab:red", "^"),
        ("flatten-tpht", "64", "tab:orange", "D"),
    ]
    phases = [("insert", "Insert"), ("lookup_hit", "Lookup (hit)")]
    lfs = sorted(lat_by_lf)

    fig, axes = plt.subplots(1, len(phases), figsize=(9.2, 3.6), sharex=True)
    for ax, (phase, title) in zip(axes, phases):
        for variant, kb, color, marker in series:
            xs, ys = [], []
            for lf in lfs:
                se = space.get((variant, kb, round(lf, 2)))
                tp = thpt.get((variant, kb, lf, phase))
                if se is None or tp is None:
                    continue
                xs.append(se)
                ys.append(tp)
            ax.plot(xs, ys, color=color, marker=marker, ms=5, lw=1.4,
                    label=f"{variant} k{kb}")
            for x, y, lf in zip(xs, ys, lfs):
                ax.annotate(f"{lf:.2f}", (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=6.5, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("space efficiency (%)")
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("throughput (Mops/s)")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("TPHT space-throughput tradeoff (16M keys, 8-byte values; "
                 "labels are load factors)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=180, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
