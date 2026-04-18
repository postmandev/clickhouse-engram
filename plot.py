"""Generate the hero chart from events ingested into ClickHouse.

Styled for a ClickHouse audience: brand palette, three panels showing
orthogonal slices of the same event stream, a header strip with query
timing, and a footer showing the actual SQL that produced the key panel.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CH_PASSWORD = os.environ["CH_PASSWORD"]
CH_USER = os.environ.get("CH_USER", "default")
CH_URL = os.environ.get("CH_URL", "http://127.0.0.1:8123/")

# ClickHouse brand palette
CH_YELLOW = "#FAFF69"
CH_YELLOW_DEEP = "#FFCC00"
CH_ORANGE = "#FFA500"
CH_BLACK = "#1F1F1F"
RED = "#D7263D"
GREY = "#BFBFBF"


def ch_query(sql: str) -> tuple[pd.DataFrame, float]:
    body = (sql + " FORMAT JSON").encode()
    req = urllib.request.Request(CH_URL, data=body, method="POST")
    auth = f"{CH_USER}:{CH_PASSWORD}".encode()
    req.add_header("Authorization", b"Basic " + base64.b64encode(auth))
    with urllib.request.urlopen(req) as resp:
        payload = json.loads(resp.read())
    elapsed_ms = payload["statistics"]["elapsed"] * 1000
    df = pd.DataFrame(payload["data"])
    return df, elapsed_ms


def main():
    hit_by_class, _ = ch_query("""
        SELECT request_class, avg(hit) AS hit_rate, count() AS lookups
        FROM engram_obs.lookups GROUP BY request_class ORDER BY hit_rate DESC
    """)
    hit_by_class["hit_rate"] = hit_by_class["hit_rate"].astype(float)

    pareto, pareto_ms = ch_query("""
        WITH per_bucket AS (
            SELECT bucket, count() AS hits
            FROM engram_obs.lookups WHERE hit = 1 GROUP BY bucket
        )
        SELECT
            row_number() OVER (ORDER BY hits DESC) / count() OVER () AS bucket_frac,
            sum(hits) OVER (ORDER BY hits DESC) / sum(hits) OVER () AS cum_hit_frac
        FROM per_bucket
        ORDER BY hits DESC
    """)
    pareto["bucket_frac"] = pareto["bucket_frac"].astype(float)
    pareto["cum_hit_frac"] = pareto["cum_hit_frac"].astype(float)

    temp_df, temp_ms = ch_query("""
        WITH per_bucket AS (
            SELECT bucket, count() AS hits
            FROM engram_obs.lookups WHERE hit = 1 GROUP BY bucket
        )
        SELECT
            countIf(hits >= 1000)               AS blazing,
            countIf(hits >= 100 AND hits < 1000) AS hot,
            countIf(hits >= 10  AND hits < 100)  AS warm,
            countIf(hits >= 1   AND hits < 10)   AS cold,
            73771 - count()                     AS dead
        FROM per_bucket
    """)
    temp = {k: int(temp_df[k].iloc[0]) for k in ["blazing", "hot", "warm", "cold", "dead"]}
    total_populated = sum(temp.values())

    totals, totals_ms = ch_query("""
        SELECT count() AS total_events,
               sum(hit) AS total_hits
        FROM engram_obs.lookups
    """)
    total_events = int(totals["total_events"].iloc[0])

    fig = plt.figure(figsize=(14, 7.6), facecolor="white")
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[0.11, 1.0, 0.22],
        hspace=0.55, wspace=0.32,
        left=0.05, right=0.97, top=0.93, bottom=0.04,
    )

    # ---------- Header strip ----------
    header = fig.add_subplot(gs[0, :])
    header.axis("off")
    header.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=header.transAxes,
        facecolor=CH_BLACK, edgecolor="none",
    ))
    header.text(
        0.015, 0.5, "ClickHouse",
        transform=header.transAxes, ha="left", va="center",
        fontsize=13, fontweight="bold", color=CH_YELLOW,
    )
    header.text(
        0.135, 0.5,
        f"│  {total_events:,} lookup events  "
        f"│  MergeTree, ORDER BY (class, layer, bucket, ts)  "
        f"│  Pareto aggregation: {pareto_ms:.0f} ms  "
        f"│  Temperature scan: {temp_ms:.0f} ms",
        transform=header.transAxes, ha="left", va="center",
        fontsize=10, color="white", family="monospace",
    )

    # ---------- Panel 1: hit rate by class ----------
    ax1 = fig.add_subplot(gs[1, 0])
    class_colors = [CH_YELLOW_DEEP, CH_YELLOW, "#FFE680", "#FFF3B8"]
    bars = ax1.bar(
        hit_by_class["request_class"],
        hit_by_class["hit_rate"] * 100,
        color=class_colors, edgecolor=CH_BLACK, linewidth=1.2,
    )
    for bar, v in zip(bars, hit_by_class["hit_rate"] * 100):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, v + 1.5,
            f"{v:.0f}%", ha="center", fontweight="bold", color=CH_BLACK, fontsize=11,
        )
    ax1.set_ylabel("Engram hit rate (%)", color=CH_BLACK)
    ax1.set_title("Hit rate varies 8× by workload", fontweight="bold", color=CH_BLACK, pad=10)
    ax1.set_ylim(0, 76)
    ax1.tick_params(axis="x", labelrotation=15, colors=CH_BLACK)
    ax1.tick_params(axis="y", colors=CH_BLACK)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color(CH_BLACK)

    # ---------- Panel 2: Pareto curve ----------
    ax2 = fig.add_subplot(gs[1, 1])
    x = pareto["bucket_frac"].to_numpy()
    y = pareto["cum_hit_frac"].to_numpy()
    ax2.fill_between(x * 100, 0, y * 100, color=CH_YELLOW, alpha=0.55, linewidth=0)
    ax2.plot(x * 100, y * 100, color=CH_BLACK, linewidth=2.2)
    ax2.plot([0, 100], [0, 100], color=GREY, linewidth=1, linestyle="--")

    def pct_at(frac: float) -> float:
        idx = int(np.argmin(np.abs(x - frac)))
        return float(y[idx] * 100)

    y1 = pct_at(0.01)
    y10 = pct_at(0.10)
    ax2.scatter([1, 10], [y1, y10], color=RED, zorder=5, s=45)
    ax2.annotate(
        f"Top 1% → {y1:.0f}% of lookups",
        xy=(1, y1), xytext=(14, y1 - 6),
        fontsize=9.5, color=CH_BLACK,
        arrowprops=dict(arrowstyle="->", color=CH_BLACK, lw=1),
    )
    ax2.annotate(
        f"Top 10% → {y10:.0f}% of lookups",
        xy=(10, y10), xytext=(26, y10 - 20),
        fontsize=9.5, color=CH_BLACK,
        arrowprops=dict(arrowstyle="->", color=CH_BLACK, lw=1),
    )
    ax2.set_xlabel("% of hit buckets (ranked by hit count)", color=CH_BLACK)
    ax2.set_ylabel("Cumulative % of lookups served", color=CH_BLACK)
    ax2.set_title("Memory access is sharply Pareto", fontweight="bold", color=CH_BLACK, pad=10)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 102)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color(CH_BLACK)
    ax2.tick_params(colors=CH_BLACK)

    # ---------- Panel 3: Temperature bar ----------
    ax3 = fig.add_subplot(gs[1, 2])
    tiers = ["blazing\n(≥1000)", "hot\n(100–999)", "warm\n(10–99)", "cold\n(1–9)", "dead\n(0)"]
    values = [temp["blazing"], temp["hot"], temp["warm"], temp["cold"], temp["dead"]]
    colors = [RED, CH_ORANGE, CH_YELLOW_DEEP, CH_YELLOW, GREY]
    pct = [100 * v / total_populated for v in values]

    y_pos = np.arange(len(tiers))
    bars3 = ax3.barh(y_pos, pct, color=colors, edgecolor=CH_BLACK, linewidth=1)
    for bar, v, p in zip(bars3, values, pct):
        ax3.text(
            max(p + 1.5, 4), bar.get_y() + bar.get_height() / 2,
            f"{v:,}  ({p:.1f}%)", va="center", fontsize=9.5, color=CH_BLACK,
        )
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(tiers, fontsize=9.5, color=CH_BLACK)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 62)
    ax3.set_xlabel("% of populated buckets", color=CH_BLACK)
    ax3.set_title(
        f"Most memorized entries are cold or dead\n"
        f"({total_populated:,} populated buckets)",
        fontweight="bold", color=CH_BLACK, pad=10,
    )
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.spines[["left", "bottom"]].set_color(CH_BLACK)
    ax3.tick_params(colors=CH_BLACK)

    # ---------- Footer SQL snippet ----------
    footer = fig.add_subplot(gs[2, :])
    footer.axis("off")
    footer.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=footer.transAxes,
        facecolor="#FAFAF5", edgecolor=CH_YELLOW_DEEP, linewidth=1.5,
    ))
    sql = (
        "WITH per_bucket AS (SELECT bucket, count() AS hits "
        "FROM engram_obs.lookups WHERE hit=1 GROUP BY bucket)\n"
        "SELECT countIf(hits>=1000) AS blazing, countIf(hits BETWEEN 100 AND 999) AS hot,\n"
        "       countIf(hits BETWEEN 10 AND 99) AS warm, countIf(hits BETWEEN 1 AND 9) AS cold,\n"
        "       73771 - count() AS dead  FROM per_bucket;"
    )
    footer.text(
        0.015, 0.5, sql,
        transform=footer.transAxes, ha="left", va="center",
        fontsize=9, family="monospace", color=CH_BLACK,
    )
    footer.text(
        0.985, 0.5, f"{temp_ms:.0f} ms",
        transform=footer.transAxes, ha="right", va="center",
        fontsize=10.5, fontweight="bold", color=CH_YELLOW_DEEP,
        family="monospace",
    )

    fig.suptitle(
        "Engram observability: per-lookup LLM telemetry in ClickHouse",
        fontsize=14.5, fontweight="bold", color=CH_BLACK, y=0.98,
    )

    out = Path("data/hero.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    print("\n--- Headline numbers ---")
    print(f"Events: {total_events:,}")
    print(f"Populated buckets: {total_populated:,}")
    for k, v in temp.items():
        print(f"  {k:<8}  {v:>6,}  ({100*v/total_populated:>5.1f}%)")
    print(f"\nPareto query: {pareto_ms:.1f} ms")
    print(f"Temperature query: {temp_ms:.1f} ms")
    print("\nHit rate by class:")
    for _, r in hit_by_class.iterrows():
        print(f"  {r['request_class']:<18}  {float(r['hit_rate'])*100:>5.1f}%")


if __name__ == "__main__":
    main()
