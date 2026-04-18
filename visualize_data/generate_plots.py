"""
Data Visualization for the LLM News Bias Analysis Project.

Generates publication-quality plots showing how the AllSides Article-Bias
dataset is structured: political-orientation proportions, source/outlet
breakdowns, topic distributions, content-length statistics, temporal
patterns, and train/valid/test split composition.

Usage (from project root):
    python visualize_data/generate_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports – add project root so configs/ and data_prep/ are importable
# Must happen BEFORE importing project modules.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from configs.config import BIAS_LABEL_MAP, BIAS_TEXT_MAP, JSONS_DIR, OUTPUT_DIR
except ImportError as e:
    raise ImportError(
        "Could not import project configs. Make sure to run from project root and have data_prep/ and configs/ directories in place."
    ) from e


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_PLOT_DIR = PROJECT_ROOT / "visualize_data"

# Consistent colour palette:  left → blue,  center → grey/purple,  right → red
BIAS_COLORS = {"left": "#3B7DD8", "center": "#7B7D7F", "right": "#D83B3B"}
BIAS_ORDER = ["left", "center", "right"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# ===================================================================
# Data loading
# ===================================================================

def load_data() -> pd.DataFrame:
    """Load articles – prefer parquet cache, fall back to raw JSONs."""
    parquet_path = Path(OUTPUT_DIR) / "articles_clean.parquet"
    if parquet_path.exists():
        print(f"Loading cached parquet from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        if "id" not in df.columns and df.index.name == "id":
            df = df.reset_index()
        return df

    # Fall back to loading from JSON files using same logic as data_prep
    print("Parquet not found – loading from raw JSON files …")
    jsons_path = Path(JSONS_DIR)
    if not jsons_path.exists():
        # Try relative to project root
        jsons_path = PROJECT_ROOT / "data" / "jsons"
    if not jsons_path.exists():
        raise FileNotFoundError(
            f"Cannot find JSON directory at {jsons_path}. "
            "Set DATASET_ROOT env var or run from project root."
        )

    json_files = sorted(jsons_path.glob("*.json"))
    print(f"  Found {len(json_files)} JSON files")

    records = []
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        article_id = data.get("ID") or data.get("id")
        title = (data.get("title") or "").strip()
        content = (data.get("content_original")
                   or data.get("content") or "").strip()
        bias_text = (data.get("bias_text") or "").strip().lower()

        if not article_id or not title or not content or bias_text not in BIAS_TEXT_MAP:
            continue

        records.append({
            "id": str(article_id).strip(),
            "topic": (data.get("topic") or "unknown").strip().lower(),
            "source": (data.get("source") or "").strip(),
            "source_url": (data.get("source_url") or "").strip(),
            "date": (data.get("date") or "").strip(),
            "authors": (data.get("authors") or "").strip(),
            "title": title,
            "content_original": content,
            "bias_text": bias_text,
            "bias": BIAS_TEXT_MAP[bias_text],
        })

    df = pd.DataFrame(records).drop_duplicates(subset="id")
    print(f"  Loaded {len(df)} valid articles")
    return df


def load_splits() -> pd.DataFrame:
    """Load train/valid/test split TSVs for both random and media splits."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    rows = []
    for split_type in ("random", "media"):
        for partition in ("train", "valid", "test"):
            tsv = splits_dir / split_type / f"{partition}.tsv"
            if not tsv.exists():
                continue
            tmp = pd.read_csv(tsv, sep="\t")
            tmp["split_type"] = split_type
            tmp["partition"] = partition
            rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by several plots."""
    df["content_len"] = df["content_original"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len()
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ===================================================================
# Plot 1 – Bias distribution pie chart
# ===================================================================

def plot_bias_pie(df: pd.DataFrame) -> None:
    counts = df["bias_text"].value_counts().reindex(BIAS_ORDER)
    colors = [BIAS_COLORS[b] for b in BIAS_ORDER]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[b.capitalize() for b in BIAS_ORDER],
        colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * counts.sum() / 100)):,})",
        startangle=140,
        textprops={"fontsize": 12},
        pctdistance=0.65,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Political Orientation Distribution\n(Full Dataset)",
                 fontsize=15, fontweight="bold")
    fig.savefig(OUTPUT_PLOT_DIR / "01_bias_distribution_pie.png")
    plt.close(fig)
    print("  ✓ 01_bias_distribution_pie.png")


# ===================================================================
# Plot 2 – Bias distribution bar chart
# ===================================================================

def plot_bias_bar(df: pd.DataFrame) -> None:
    counts = df["bias_text"].value_counts().reindex(BIAS_ORDER)
    total = counts.sum()
    colors = [BIAS_COLORS[b] for b in BIAS_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([b.capitalize() for b in BIAS_ORDER], counts, color=colors,
                  edgecolor="white", linewidth=1.2, width=0.55)

    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Number of Articles")
    ax.set_title("Article Count by Political Orientation", fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_PLOT_DIR / "02_bias_distribution_bar.png")
    plt.close(fig)
    print("  ✓ 02_bias_distribution_bar.png")


# ===================================================================
# Plot 3 – Bias distribution by topic (stacked bar)
# ===================================================================

def plot_bias_by_topic(df: pd.DataFrame) -> None:
    ct = pd.crosstab(df["topic"], df["bias_text"]).reindex(
        columns=BIAS_ORDER, fill_value=0)
    ct = ct.sort_values(by=BIAS_ORDER, ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(6, len(ct) * 0.45)))
    ct.plot.barh(stacked=True, color=[BIAS_COLORS[b] for b in BIAS_ORDER], ax=ax,
                 edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Number of Articles")
    ax.set_ylabel("")
    ax.set_title("Political Orientation Distribution by Topic",
                 fontweight="bold")
    ax.legend(title="Bias", labels=[b.capitalize() for b in BIAS_ORDER])
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    fig.savefig(OUTPUT_PLOT_DIR / "03_bias_by_topic.png")
    plt.close(fig)
    print("  ✓ 03_bias_by_topic.png")


# ===================================================================
# Plot 4 – Bias proportion by topic (100% stacked bar)
# ===================================================================

def plot_bias_proportion_by_topic(df: pd.DataFrame) -> None:
    ct = pd.crosstab(df["topic"], df["bias_text"]).reindex(
        columns=BIAS_ORDER, fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.sort_values(by="center", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(ct_pct) * 0.45)))
    ct_pct.plot.barh(stacked=True, color=[BIAS_COLORS[b] for b in BIAS_ORDER], ax=ax,
                     edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Percentage of Articles (%)")
    ax.set_ylabel("")
    ax.set_title(
        "Political Orientation Proportion by Topic (Normalized)", fontweight="bold")
    ax.legend(title="Bias", labels=[b.capitalize() for b in BIAS_ORDER])
    ax.set_xlim(0, 100)
    ax.axvline(50, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    fig.savefig(OUTPUT_PLOT_DIR / "04_bias_proportion_by_topic.png")
    plt.close(fig)
    print("  ✓ 04_bias_proportion_by_topic.png")


# ===================================================================
# Plot 5 – Top 20 outlets by article count (coloured by majority bias)
# ===================================================================

def plot_top_outlets(df: pd.DataFrame, top_n: int = 20) -> None:
    source_counts = df["source"].value_counts().head(top_n)
    # Determine majority bias for each outlet
    majority_bias = (
        df[df["source"].isin(source_counts.index)]
        .groupby("source")["bias_text"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    colors = [BIAS_COLORS[majority_bias.get(
        s, "center")] for s in source_counts.index]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.38)))
    bars = ax.barh(source_counts.index[::-1], source_counts.values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, source_counts.values[::-1]):
        ax.text(bar.get_width() + source_counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    ax.set_xlabel("Number of Articles")
    ax.set_title(f"Top {top_n} News Outlets by Article Count\n(coloured by majority political leaning)",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BIAS_COLORS[b], label=b.capitalize())
               for b in BIAS_ORDER]
    ax.legend(handles=handles, title="Majority Bias", loc="lower right")

    fig.savefig(OUTPUT_PLOT_DIR / "05_top_outlets.png")
    plt.close(fig)
    print("  ✓ 05_top_outlets.png")


# ===================================================================
# Plot 6 – Unique outlet count per bias category
# ===================================================================

def plot_outlets_per_bias(df: pd.DataFrame) -> None:
    # Assign each outlet its majority bias
    majority = df.groupby("source")["bias_text"].agg(
        lambda s: s.value_counts().idxmax())
    outlet_bias_counts = majority.value_counts().reindex(BIAS_ORDER, fill_value=0)
    colors = [BIAS_COLORS[b] for b in BIAS_ORDER]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([b.capitalize() for b in BIAS_ORDER], outlet_bias_counts, color=colors,
                  edgecolor="white", linewidth=1.2, width=0.5)

    for bar, val in zip(bars, outlet_bias_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Number of Unique Outlets")
    ax.set_title(
        "Number of Unique News Outlets per Political Leaning", fontweight="bold")
    ax.set_ylim(0, outlet_bias_counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_PLOT_DIR / "06_outlets_per_bias.png")
    plt.close(fig)
    print("  ✓ 06_outlets_per_bias.png")


# ===================================================================
# Plot 7 – Outlet × bias heatmap (top 15 outlets)
# ===================================================================

def plot_outlet_bias_heatmap(df: pd.DataFrame, top_n: int = 15) -> None:
    top_sources = df["source"].value_counts().head(top_n).index
    sub = df[df["source"].isin(top_sources)]
    ct = pd.crosstab(sub["source"], sub["bias_text"]).reindex(
        columns=BIAS_ORDER, fill_value=0)
    ct = ct.loc[top_sources]  # preserve rank order

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.4)))
    im = ax.imshow(ct.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(BIAS_ORDER)))
    ax.set_xticklabels([b.capitalize() for b in BIAS_ORDER])
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels(ct.index)

    # Annotate cells
    for i in range(len(ct)):
        for j in range(len(BIAS_ORDER)):
            val = ct.values[i, j]
            text_color = "white" if val > ct.values.max() * 0.6 else "black"
            ax.text(j, i, f"{val:,}", ha="center",
                    va="center", fontsize=10, color=text_color)

    ax.set_title(
        f"Article Count: Top {top_n} Outlets × Political Orientation", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Article Count", shrink=0.8)
    fig.savefig(OUTPUT_PLOT_DIR / "07_outlet_bias_heatmap.png")
    plt.close(fig)
    print("  ✓ 07_outlet_bias_heatmap.png")


# ===================================================================
# Plot 8 – Topic distribution bar chart
# ===================================================================

def plot_topic_distribution(df: pd.DataFrame) -> None:
    topic_counts = df["topic"].value_counts()

    fig, ax = plt.subplots(figsize=(12, max(5, len(topic_counts) * 0.4)))
    bars = ax.barh(topic_counts.index[::-1], topic_counts.values[::-1],
                   color="#4A90D9", edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, topic_counts.values[::-1]):
        ax.text(bar.get_width() + topic_counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    ax.set_xlabel("Number of Articles")
    ax.set_title("Article Count by Topic", fontweight="bold")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_PLOT_DIR / "08_topic_distribution.png")
    plt.close(fig)
    print("  ✓ 08_topic_distribution.png")


# ===================================================================
# Plot 9 – Topic × bias heatmap
# ===================================================================

def plot_topic_bias_heatmap(df: pd.DataFrame) -> None:
    ct = pd.crosstab(df["topic"], df["bias_text"]).reindex(
        columns=BIAS_ORDER, fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(8, max(5, len(ct) * 0.42)))
    im = ax.imshow(ct.values, aspect="auto", cmap="BuPu")

    ax.set_xticks(range(len(BIAS_ORDER)))
    ax.set_xticklabels([b.capitalize() for b in BIAS_ORDER])
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels(ct.index)

    for i in range(len(ct)):
        for j in range(len(BIAS_ORDER)):
            val = ct.values[i, j]
            text_color = "white" if val > ct.values.max() * 0.55 else "black"
            ax.text(j, i, f"{val:,}", ha="center",
                    va="center", fontsize=10, color=text_color)

    ax.set_title("Topic × Political Orientation Heatmap", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Article Count", shrink=0.8)
    fig.savefig(OUTPUT_PLOT_DIR / "09_topic_bias_heatmap.png")
    plt.close(fig)
    print("  ✓ 09_topic_bias_heatmap.png")


# ===================================================================
# Plot 10 – Article content-length distribution by bias
# ===================================================================

def plot_content_length_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for bias in BIAS_ORDER:
        subset = df.loc[df["bias_text"] == bias, "content_len"].dropna()
        ax.hist(subset, bins=80, alpha=0.55, label=bias.capitalize(),
                color=BIAS_COLORS[bias], edgecolor="none")

    ax.set_xlabel("Article Content Length (characters)")
    ax.set_ylabel("Number of Articles")
    ax.set_title(
        "Distribution of Article Content Length by Political Orientation", fontweight="bold")
    ax.legend(title="Bias")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)

    # Add vertical lines for medians
    for bias in BIAS_ORDER:
        med = df.loc[df["bias_text"] == bias, "content_len"].median()
        ax.axvline(med, color=BIAS_COLORS[bias],
                   linestyle="--", linewidth=1.2, alpha=0.8)

    fig.savefig(OUTPUT_PLOT_DIR / "10_content_length_distribution.png")
    plt.close(fig)
    print("  ✓ 10_content_length_distribution.png")


# ===================================================================
# Plot 11 – Article content length by topic (box plot)
# ===================================================================

def plot_content_length_by_topic(df: pd.DataFrame) -> None:
    topic_order = df.groupby("topic")["content_len"].median(
    ).sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(12, max(5, len(topic_order) * 0.42)))
    data_groups = [df.loc[df["topic"] == t,
                          "content_len"].dropna().values for t in topic_order]
    bp = ax.boxplot(data_groups, vert=False, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor="#4A90D9", alpha=0.6),
                    medianprops=dict(color="black", linewidth=1.5))

    ax.set_yticklabels(topic_order)
    ax.set_xlabel("Article Content Length (characters)")
    ax.set_title("Article Content Length by Topic", fontweight="bold")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_PLOT_DIR / "11_content_length_by_topic.png")
    plt.close(fig)
    print("  ✓ 11_content_length_by_topic.png")


# ===================================================================
# Plot 12 – Title word-count distribution by bias
# ===================================================================

def plot_title_length_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for bias in BIAS_ORDER:
        subset = df.loc[df["bias_text"] == bias, "title_word_count"].dropna()
        ax.hist(subset, bins=40, alpha=0.55, label=bias.capitalize(),
                color=BIAS_COLORS[bias], edgecolor="none")

    ax.set_xlabel("Title Length (word count)")
    ax.set_ylabel("Number of Articles")
    ax.set_title(
        "Distribution of Title Length by Political Orientation", fontweight="bold")
    ax.legend(title="Bias")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_PLOT_DIR / "12_title_length_distribution.png")
    plt.close(fig)
    print("  ✓ 12_title_length_distribution.png")


# ===================================================================
# Plot 13 – Articles over time (monthly)
# ===================================================================

def plot_articles_over_time(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["date_parsed"]).copy()
    if valid.empty:
        print("  ⚠ Skipping temporal plot – no parseable dates")
        return

    valid["year_month"] = valid["date_parsed"].dt.to_period("M")
    monthly = valid.groupby("year_month").size()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(monthly.index.astype(str), monthly.values,
            color="#4A90D9", linewidth=1.5)
    ax.fill_between(range(len(monthly)), monthly.values,
                    alpha=0.15, color="#4A90D9")

    # Thin out x-tick labels to avoid overlap
    tick_step = max(1, len(monthly) // 20)
    ax.set_xticks(range(0, len(monthly), tick_step))
    ax.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), tick_step)],
                       rotation=45, ha="right", fontsize=9)

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Article Publication Volume Over Time", fontweight="bold")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_PLOT_DIR / "13_articles_over_time.png")
    plt.close(fig)
    print("  ✓ 13_articles_over_time.png")


# ===================================================================
# Plot 14 – Bias distribution over time (stacked area)
# ===================================================================

def plot_bias_over_time(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["date_parsed"]).copy()
    if valid.empty:
        print("  ⚠ Skipping temporal-bias plot – no parseable dates")
        return

    valid["year_month"] = valid["date_parsed"].dt.to_period("M")
    ct = pd.crosstab(valid["year_month"], valid["bias_text"]).reindex(
        columns=BIAS_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = range(len(ct))
    ax.stackplot(x, *[ct[b].values for b in BIAS_ORDER],
                 labels=[b.capitalize() for b in BIAS_ORDER],
                 colors=[BIAS_COLORS[b] for b in BIAS_ORDER], alpha=0.75)

    tick_step = max(1, len(ct) // 20)
    ax.set_xticks(range(0, len(ct), tick_step))
    ax.set_xticklabels([str(ct.index[i]) for i in range(0, len(ct), tick_step)],
                       rotation=45, ha="right", fontsize=9)

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Political Orientation Distribution Over Time",
                 fontweight="bold")
    ax.legend(title="Bias", loc="upper left")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_PLOT_DIR / "14_bias_over_time.png")
    plt.close(fig)
    print("  ✓ 14_bias_over_time.png")


# ===================================================================
# Plot 15 – Data-split composition (random & media)
# ===================================================================

def plot_split_composition(splits_df: pd.DataFrame) -> None:
    if splits_df.empty:
        print("  ⚠ Skipping split plot – no split data found")
        return

    summary = splits_df.groupby(
        ["split_type", "partition"]).size().unstack(fill_value=0)
    partition_order = ["train", "valid", "test"]
    summary = summary.reindex(
        columns=[p for p in partition_order if p in summary.columns])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    width = 0.22
    part_colors = {"train": "#4A90D9", "valid": "#F5A623", "test": "#7ED321"}

    for i, part in enumerate(summary.columns):
        bars = ax.bar(x + i * width, summary[part], width, label=part.capitalize(),
                      color=part_colors.get(part, "#999"), edgecolor="white", linewidth=1)
        for bar, val in zip(bars, summary[part]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + summary.values.max() * 0.01,
                    f"{val:,}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels([s.capitalize() for s in summary.index])
    ax.set_ylabel("Number of Articles")
    ax.set_title("Train / Validation / Test Split Sizes", fontweight="bold")
    ax.legend(title="Partition")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_PLOT_DIR / "15_split_composition.png")
    plt.close(fig)
    print("  ✓ 15_split_composition.png")


# ===================================================================
# Summary statistics table (saved as image)
# ===================================================================

def plot_summary_table(df: pd.DataFrame) -> None:
    total = len(df)
    bias_counts = df["bias_text"].value_counts().reindex(
        BIAS_ORDER, fill_value=0)
    stats = {
        "Total articles": f"{total:,}",
        "Unique topics": f"{df['topic'].nunique()}",
        "Unique outlets": f"{df['source'].nunique()}",
        "Left articles": f"{bias_counts['left']:,}  ({100*bias_counts['left']/total:.1f}%)",
        "Center articles": f"{bias_counts['center']:,}  ({100*bias_counts['center']/total:.1f}%)",
        "Right articles": f"{bias_counts['right']:,}  ({100*bias_counts['right']/total:.1f}%)",
        "Median content length": f"{df['content_len'].median():,.0f} chars",
        "Date range": f"{df['date_parsed'].min():%Y-%m-%d}  –  {df['date_parsed'].max():%Y-%m-%d}"
        if df["date_parsed"].notna().any() else "N/A",
    }

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    table = ax.table(
        cellText=[[k, v] for k, v in stats.items()],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header row
    for j in range(2):
        table[0, j].set_facecolor("#4A90D9")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Dataset Summary Statistics",
                 fontweight="bold", fontsize=14, pad=20)
    fig.savefig(OUTPUT_PLOT_DIR / "00_summary_table.png")
    plt.close(fig)
    print("  ✓ 00_summary_table.png")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

    print("=" * 60)
    print("  DATA VISUALIZATION – LLM News Bias Analysis")
    print("=" * 60)

    # ── Load & enrich ────────────────────────────────────────────
    df = load_data()
    df = enrich(df)
    print(
        f"\nDataset: {len(df):,} articles | {df['topic'].nunique()} topics | {df['source'].nunique()} outlets\n")

    # ── Generate plots ───────────────────────────────────────────
    print("Generating plots …")

    plot_summary_table(df)

    # Political orientation (core requirement)
    plot_bias_pie(df)
    plot_bias_bar(df)
    plot_bias_by_topic(df)
    plot_bias_proportion_by_topic(df)

    # Source / outlet analysis
    plot_top_outlets(df)
    plot_outlets_per_bias(df)
    plot_outlet_bias_heatmap(df)

    # Topic analysis
    plot_topic_distribution(df)
    plot_topic_bias_heatmap(df)

    # Content statistics
    plot_content_length_distribution(df)
    plot_content_length_by_topic(df)
    plot_title_length_distribution(df)

    # Temporal analysis
    plot_articles_over_time(df)
    plot_bias_over_time(df)

    # Data splits
    splits_df = load_splits()
    plot_split_composition(splits_df)

    print(f"\nAll plots saved to: {OUTPUT_PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
