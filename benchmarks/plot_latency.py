#!/usr/bin/env python3
"""Plot the TPHT latency sweep produced by benchmarks/tpht_latency.c.

Usage: plot_latency.py results/tpht_latency.csv [output_dir]

Writes tpht_latency.pdf next to the CSV.
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
# Embed TrueType rather than Type 3 outlines: Type 3 is rejected by several
# paper submission systems and keeps the text unselectable.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

VARIANTS = ["chained-tpht", "flatten-tpht"]
VARIANT_LABEL = {"chained-tpht": "chained", "flatten-tpht": "flattened"}
PHASES = [
    ("insert", "Insert"),
    ("lookup_hit", "Lookup, present"),
    ("lookup_miss", "Lookup, absent"),
    ("remove", "Remove"),
]

# Validated categorical slots 1 and 2.
THEME = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text2": "#52514e",
    "grid": "#e4e3df",
    "series": {"chained-tpht": "#2a78d6", "flatten-tpht": "#eb6834"},
}

def si(n):
    if n >= 1 << 20:
        return "%gM" % (n / (1 << 20))
    if n >= 1 << 10:
        return "%gK" % (n / (1 << 10))
    return str(int(n))


def load(path):
    """rows[key_bits][phase][variant] -> sorted [(keys, ns_per_op), ...]"""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            rows[int(r["key_bits"])][r["phase"]][r["variant"]].append(
                (int(r["keys"]), float(r["ns_per_op"]))
            )
    for kb in rows:
        for ph in rows[kb]:
            for v in rows[kb][ph]:
                rows[kb][ph][v].sort()
    return rows


def draw(rows, out_path, subtitle):
    th = THEME
    key_bits = sorted(rows)
    fig, axes = plt.subplots(
        len(key_bits),
        len(PHASES),
        figsize=(14.5, 3.5 * len(key_bits) + 1.4),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    fig.patch.set_facecolor(th["surface"])

    for row, kb in enumerate(key_bits):
        row_max = 0.0
        for col, (phase, phase_label) in enumerate(PHASES):
            ax = axes[row][col]
            ax.set_facecolor(th["surface"])
            ends = []

            for variant in VARIANTS:
                pts = rows[kb][phase].get(variant, [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                colour = th["series"][variant]
                ax.plot(
                    xs,
                    ys,
                    color=colour,
                    linewidth=2,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    marker="o",
                    markersize=5,
                    markeredgewidth=1,
                    markeredgecolor=th["surface"],
                    zorder=3,
                )
                ends.append((ys[-1], xs[-1], variant))
                row_max = max(row_max, max(ys))

            # Direct-label the endpoint only; the axis carries everything else.
            ends.sort()
            prev = None
            for value, x, _variant in ends:
                offset = 6
                if prev is not None and abs(value - prev) < 0.06 * max(
                    1.0, ax.get_ylim()[1]
                ):
                    offset = 14
                prev = value
                ax.annotate(
                    "%.0f ns" % value,
                    xy=(x, value),
                    xytext=(7, offset - 6),
                    textcoords="offset points",
                    color=th["text2"],
                    fontsize=8.5,
                    va="center",
                    clip_on=False,
                    zorder=4,
                )

            ax.set_xscale("log", base=2)
            ax.grid(True, which="major", color=th["grid"], linewidth=0.8, linestyle="-")
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(th["grid"])
                ax.spines[side].set_linewidth(0.8)
            ax.tick_params(colors=th["text2"], labelsize=9, length=0)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: si(v)))
            ax.margins(x=0.20)

            if row == 0:
                ax.set_title(phase_label, color=th["text"], fontsize=11, pad=10)
            if row == len(key_bits) - 1:
                ax.set_xlabel("keys in table", color=th["text2"], fontsize=9.5)
            if col == 0:
                ax.set_ylabel(
                    "%d-bit keys\nns / operation" % kb, color=th["text2"], fontsize=9.5
                )

        # One shared scale per row, sized to the row's own maximum plus room
        # for the endpoint labels.
        axes[row][0].set_ylim(0, row_max * 1.12)

    handles = [
        Line2D(
            [],
            [],
            color=th["series"][v],
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgewidth=1,
            markeredgecolor=th["surface"],
            label=VARIANT_LABEL[v],
        )
        for v in VARIANTS
    ]
    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.062, 0.918),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color(th["text"])

    fig.suptitle(
        "TPHT operation latency vs table size",
        x=0.062,
        y=0.988,
        ha="left",
        color=th["text"],
        fontsize=15,
    )
    fig.text(0.062, 0.951, subtitle, ha="left", color=th["text2"], fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.985, 0.885))

    fig.savefig(
        out_path,
        facecolor=th["surface"],
        metadata={"Title": "TPHT operation latency vs table size"},
    )
    print("[tpht] wrote %s" % out_path)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path) or "."
    rows = load(csv_path)

    with open(csv_path, newline="") as handle:
        first = next(csv.DictReader(handle))
    subtitle = (
        "sequential, fixed-capacity, %s-byte values, load factor %.2f  ·  "
        "best of repeated runs, single thread"
        % (int(first["value_bits"]) // 8, float(first["load_factor"]))
    )

    draw(rows, os.path.join(out_dir, "tpht_latency.pdf"), subtitle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
