# analysis/analyze_logs.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_sources(row: Dict[str, Any]) -> Tuple[int, int]:
    """
    Returns:
      - sources_count: number of sources returned
      - unique_pages_count: number of unique (source,page) pairs
    """
    sources = row.get("sources", []) or []
    sources_count = len(sources)
    unique = set()
    for s in sources:
        src = s.get("source")
        page = s.get("page")
        unique.add((src, page))
    return sources_count, len(unique)


def build_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(rows)

    # Ensure expected columns exist
    for col in ["latency_ms", "grounded", "disclosure_mode", "question", "ui", "error"]:
        if col not in df.columns:
            df[col] = None

    # Clean latency
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")

    # Grounded as boolean
    df["grounded"] = df["grounded"].fillna(False).astype(bool)

    # Sources features
    src_counts = []
    uniq_counts = []
    for r in rows:
        sc, uc = normalize_sources(r)
        src_counts.append(sc)
        uniq_counts.append(uc)
    df["sources_count"] = src_counts
    df["unique_source_pages"] = uniq_counts

    # Error flag
    df["has_error"] = df["error"].notna() & (df["error"].astype(str).str.len() > 0)

    return df


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    df_ok = df[~df["has_error"]].copy()

    out["n_total"] = int(len(df))
    out["n_ok"] = int(len(df_ok))
    out["n_error"] = int(df["has_error"].sum())

    if len(df_ok) > 0:
        out["latency_ms_mean"] = float(df_ok["latency_ms"].mean())
        out["latency_ms_median"] = float(df_ok["latency_ms"].median())
        out["latency_ms_p95"] = float(df_ok["latency_ms"].quantile(0.95))
        out["grounded_rate"] = float(df_ok["grounded"].mean())
        out["avg_sources_count"] = float(df_ok["sources_count"].mean())
        out["avg_unique_source_pages"] = float(df_ok["unique_source_pages"].mean())
    else:
        out["latency_ms_mean"] = None
        out["latency_ms_median"] = None
        out["latency_ms_p95"] = None
        out["grounded_rate"] = None
        out["avg_sources_count"] = None
        out["avg_unique_source_pages"] = None

    # Breakdown tables
    out["by_mode"] = (
        df_ok.groupby("disclosure_mode")
        .agg(
            n=("question", "count"),
            grounded_rate=("grounded", "mean"),
            latency_mean=("latency_ms", "mean"),
            latency_median=("latency_ms", "median"),
            sources_mean=("sources_count", "mean"),
        )
        .sort_values("n", ascending=False)
        .reset_index()
        .to_dict(orient="records")
        if len(df_ok) > 0
        else []
    )

    out["by_ui"] = (
        df_ok.groupby("ui")
        .agg(
            n=("question", "count"),
            grounded_rate=("grounded", "mean"),
            latency_mean=("latency_ms", "mean"),
        )
        .sort_values("n", ascending=False)
        .reset_index()
        .to_dict(orient="records")
        if len(df_ok) > 0
        else []
    )

    return out


def save_summary(summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "rows.csv", index=False)

    df_ok = df[~df["has_error"]].copy()

    # Tables useful for paper
    by_mode = (
        df_ok.groupby("disclosure_mode")
        .agg(
            n=("question", "count"),
            grounded_rate=("grounded", "mean"),
            latency_mean=("latency_ms", "mean"),
            latency_median=("latency_ms", "median"),
            latency_p95=("latency_ms", lambda x: x.quantile(0.95)),
            sources_mean=("sources_count", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    by_mode.to_csv(out_dir / "by_mode.csv", index=False)

    by_ui = (
        df_ok.groupby("ui")
        .agg(
            n=("question", "count"),
            grounded_rate=("grounded", "mean"),
            latency_mean=("latency_ms", "mean"),
            latency_median=("latency_ms", "median"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    by_ui.to_csv(out_dir / "by_ui.csv", index=False)


def plot_latency_hist(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ok = df[~df["has_error"]].copy()
    df_ok = df_ok[df_ok["latency_ms"].notna()]

    if len(df_ok) == 0:
        return

    plt.figure()
    plt.hist(df_ok["latency_ms"], bins=20)
    plt.title("Latency Distribution (ms)")
    plt.xlabel("latency_ms")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_hist.png", dpi=200)
    plt.close()


def plot_latency_by_mode(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ok = df[~df["has_error"]].copy()
    df_ok = df_ok[df_ok["latency_ms"].notna()]

    if len(df_ok) == 0:
        return

    modes = sorted([m for m in df_ok["disclosure_mode"].dropna().unique()])
    if not modes:
        return

    data = [df_ok[df_ok["disclosure_mode"] == m]["latency_ms"].values for m in modes]

    plt.figure()
    plt.boxplot(data, labels=modes, showfliers=False)
    plt.title("Latency by Disclosure Mode (ms)")
    plt.xlabel("disclosure_mode")
    plt.ylabel("latency_ms")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_by_mode_boxplot.png", dpi=200)
    plt.close()


def plot_grounded_rate_by_mode(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ok = df[~df["has_error"]].copy()

    if len(df_ok) == 0 or df_ok["disclosure_mode"].isna().all():
        return

    g = (
        df_ok.groupby("disclosure_mode")["grounded"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure()
    plt.bar(g.index.astype(str), g.values)
    plt.title("Grounded Rate by Disclosure Mode")
    plt.xlabel("disclosure_mode")
    plt.ylabel("grounded_rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "grounded_rate_by_mode.png", dpi=200)
    plt.close()


def plot_sources_count(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ok = df[~df["has_error"]].copy()

    if len(df_ok) == 0:
        return

    plt.figure()
    plt.hist(df_ok["sources_count"], bins=range(0, int(df_ok["sources_count"].max() + 2)))
    plt.title("Sources Count Distribution")
    plt.xlabel("sources_count")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "sources_count_hist.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=str,
        default="logs/eval.jsonl",
        help="Path to a jsonl log file (e.g., logs/eval.jsonl or logs/audit.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="analysis/out",
        help="Output directory for summary/tables/plots",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out)

    rows = read_jsonl(log_path)
    df = build_dataframe(rows)

    summary = summarize(df)
    save_summary(summary, out_dir)
    save_tables(df, out_dir)

    plot_latency_hist(df, out_dir)
    plot_latency_by_mode(df, out_dir)
    plot_grounded_rate_by_mode(df, out_dir)
    plot_sources_count(df, out_dir)

    # Print a quick console summary
    print(f"Loaded: {log_path} ({len(df)} rows)")
    print(f"Output: {out_dir}")
    print(f"OK: {summary['n_ok']}, Errors: {summary['n_error']}")
    if summary["latency_ms_mean"] is not None:
        print(
            "Latency ms (mean/median/p95): "
            f"{summary['latency_ms_mean']:.1f} / {summary['latency_ms_median']:.1f} / {summary['latency_ms_p95']:.1f}"
        )
        print(f"Grounded rate: {summary['grounded_rate']:.3f}")
        print(f"Avg sources: {summary['avg_sources_count']:.2f}")


    print(f"Working directory: {Path.cwd()}")
    print(f"Writing outputs to: {out_dir.resolve()}")



if __name__ == "__main__":
    main()
