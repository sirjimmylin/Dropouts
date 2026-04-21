#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


EUROSAT_PAPER_TARGETS = {
    ("linear", "resnet18"): (93.08, 94.72),
    ("linear", "resnet50"): (94.48, 95.90),
    ("finetune", "resnet18"): (97.02, 97.47),
    ("finetune", "resnet50"): (97.17, 97.77),
}

BIGEARTHNET_PAPER_TARGETS = {
    ("linear", "resnet18", 0.1): (67.89, 68.64),
    ("linear", "resnet18", 1.0): (69.43, 70.41),
    ("linear", "resnet50", 0.1): (71.55, 73.40),
    ("linear", "resnet50", 1.0): (73.63, 74.98),
}


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def canonical_train_frac(value: Any) -> float:
    numeric = safe_float(value)
    if numeric is None:
        return 1.0
    return round(numeric, 4)


def resolve_artifact_path(run_dir: Path, raw_path: Any) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    from_run_dir = run_dir / candidate
    if from_run_dir.exists():
        return from_run_dir

    from_cwd = Path.cwd() / candidate
    if from_cwd.exists():
        return from_cwd

    return from_run_dir


def read_last_metrics_row(metrics_csv: Path | None) -> dict[str, str]:
    if metrics_csv is None or not metrics_csv.exists():
        return {}

    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return rows[-1]


def load_run_summaries(results_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(results_root.rglob("summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dataset = summary.get("dataset")
        if dataset not in {"EuroSAT", "BigEarthNet"}:
            continue

        metric_key = "val_acc" if dataset == "EuroSAT" else "val_map"
        best_key = "best_val_acc" if dataset == "EuroSAT" else "best_val_map"
        final_key = "final_val_acc" if dataset == "EuroSAT" else "final_val_map"

        metrics_csv = resolve_artifact_path(run_dir, summary.get("metrics_csv"))
        last_metrics_row = read_last_metrics_row(metrics_csv)
        final_metric = safe_float(summary.get(final_key))
        if final_metric is None:
            final_metric = safe_float(last_metrics_row.get(metric_key))

        summaries.append(
            {
                "dataset": dataset,
                "run_name": run_dir.name,
                "base_encoder": summary.get("base_encoder", ""),
                "train_mode": summary.get("train_mode", ""),
                "backbone_type": summary.get("backbone_type", ""),
                "seed": summary.get("seed"),
                "train_frac": canonical_train_frac(summary.get("train_frac", 1.0)),
                "best_metric": safe_float(summary.get(best_key)),
                "best_epoch": summary.get("best_epoch"),
                "final_metric": final_metric,
                "summary_path": str(summary_path),
                "metrics_csv": str(metrics_csv) if metrics_csv is not None else "",
            }
        )
    return summaries


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = (
            run["dataset"],
            run["base_encoder"],
            run["train_mode"],
            run["backbone_type"],
            run["train_frac"],
        )
        grouped[key].append(run)

    rows: list[dict[str, Any]] = []
    for key, group_runs in sorted(grouped.items()):
        best_values = [x["best_metric"] for x in group_runs if x["best_metric"] is not None]
        final_values = [x["final_metric"] for x in group_runs if x["final_metric"] is not None]
        seed_values = sorted({str(x["seed"]) for x in group_runs if x["seed"] is not None})

        mean_best = mean(best_values) if best_values else None
        std_best = stdev(best_values) if len(best_values) > 1 else (0.0 if best_values else None)
        mean_final = mean(final_values) if final_values else None
        std_final = stdev(final_values) if len(final_values) > 1 else (0.0 if final_values else None)

        rows.append(
            {
                "dataset": key[0],
                "base_encoder": key[1],
                "train_mode": key[2],
                "backbone_type": key[3],
                "train_frac": key[4],
                "n_runs": len(group_runs),
                "seeds": ",".join(seed_values),
                "mean_best_metric": mean_best,
                "std_best_metric": std_best,
                "mean_final_metric": mean_final,
                "std_final_metric": std_final,
            }
        )
    return rows


def build_paper_comparison_rows(aggregated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    for row in aggregated_rows:
        dataset = row["dataset"]
        train_mode = row["train_mode"]
        base_encoder = row["base_encoder"]
        train_frac = canonical_train_frac(row["train_frac"])
        ours_best = row["mean_best_metric"]
        paper_100k = None
        paper_1m = None

        if dataset == "EuroSAT":
            paper_100k, paper_1m = EUROSAT_PAPER_TARGETS.get((train_mode, base_encoder), (None, None))
        elif dataset == "BigEarthNet":
            paper_100k, paper_1m = BIGEARTHNET_PAPER_TARGETS.get((train_mode, base_encoder, train_frac), (None, None))

        delta_100k = None
        delta_1m = None
        if row["backbone_type"] == "pretrain" and ours_best is not None and paper_100k is not None:
            delta_100k = ours_best - paper_100k
        if row["backbone_type"] == "pretrain" and ours_best is not None and paper_1m is not None:
            delta_1m = ours_best - paper_1m

        if row["backbone_type"] == "pretrain":
            note = "Our pretrain checkpoint/setup may differ from paper-reported CACo-100k/1m settings."
        else:
            note = "Baseline initialization in our setup."

        comparison_rows.append(
            {
                "dataset": dataset,
                "base_encoder": base_encoder,
                "train_mode": train_mode,
                "train_frac": train_frac,
                "backbone_type": row["backbone_type"],
                "n_runs": row["n_runs"],
                "seeds": row["seeds"],
                "ours_mean_best": ours_best,
                "ours_std_best": row["std_best_metric"],
                "ours_mean_final": row["mean_final_metric"],
                "paper_caco_100k": paper_100k,
                "paper_caco_1m": paper_1m,
                "delta_vs_paper_100k": delta_100k,
                "delta_vs_paper_1m": delta_1m,
                "notes": note,
            }
        )
    return comparison_rows


def build_claim_check_rows(aggregated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[Any, ...], float] = {}
    for row in aggregated_rows:
        metric = row.get("mean_best_metric")
        if metric is None:
            continue
        key = (
            row["dataset"],
            row["base_encoder"],
            row["train_mode"],
            row["backbone_type"],
            canonical_train_frac(row["train_frac"]),
        )
        index[key] = metric

    claim_rows: list[dict[str, Any]] = []

    def get_metric(dataset: str, base_encoder: str, train_mode: str, backbone_type: str, train_frac: float) -> float | None:
        return index.get((dataset, base_encoder, train_mode, backbone_type, canonical_train_frac(train_frac)))

    datasets = sorted({row["dataset"] for row in aggregated_rows})
    base_encoders = sorted({row["base_encoder"] for row in aggregated_rows})
    train_fracs = sorted({canonical_train_frac(row["train_frac"]) for row in aggregated_rows})

    for base_encoder in base_encoders:
        pre = get_metric("EuroSAT", base_encoder, "linear", "pretrain", 1.0)
        random_init = get_metric("EuroSAT", base_encoder, "linear", "random", 1.0)
        imagenet = get_metric("EuroSAT", base_encoder, "linear", "imagenet", 1.0)
        if pre is None or random_init is None or imagenet is None:
            status = "insufficient_data"
            details = "Need EuroSAT linear runs for random, imagenet, pretrain."
        else:
            strongest_baseline = max(random_init, imagenet)
            status = "pass" if pre > strongest_baseline else "fail"
            details = (
                f"pretrain={pre:.3f}, imagenet={imagenet:.3f}, random={random_init:.3f}, "
                f"margin={pre - strongest_baseline:.3f}"
            )
        claim_rows.append(
            {
                "claim_id": f"claim-1-eurosat-linear-{base_encoder}",
                "category": "claim_verification",
                "status": status,
                "dataset": "EuroSAT",
                "base_encoder": base_encoder,
                "train_mode": "linear",
                "train_frac": 1.0,
                "details": details,
            }
        )

    for base_encoder in base_encoders:
        for train_frac in train_fracs:
            if train_frac not in {0.1, 1.0}:
                continue
            pre = get_metric("BigEarthNet", base_encoder, "linear", "pretrain", train_frac)
            random_init = get_metric("BigEarthNet", base_encoder, "linear", "random", train_frac)
            imagenet = get_metric("BigEarthNet", base_encoder, "linear", "imagenet", train_frac)
            if pre is None or random_init is None or imagenet is None:
                status = "insufficient_data"
                details = "Need BigEarthNet linear runs for random, imagenet, pretrain."
            else:
                strongest_baseline = max(random_init, imagenet)
                status = "pass" if pre > strongest_baseline else "fail"
                details = (
                    f"pretrain={pre:.3f}, imagenet={imagenet:.3f}, random={random_init:.3f}, "
                    f"margin={pre - strongest_baseline:.3f}"
                )
            claim_rows.append(
                {
                    "claim_id": f"claim-2-bigearthnet-linear-{base_encoder}-frac{train_frac:g}",
                    "category": "claim_verification",
                    "status": status,
                    "dataset": "BigEarthNet",
                    "base_encoder": base_encoder,
                    "train_mode": "linear",
                    "train_frac": train_frac,
                    "details": details,
                }
            )

    for base_encoder in base_encoders:
        low_pre = get_metric("BigEarthNet", base_encoder, "linear", "pretrain", 0.1)
        low_imagenet = get_metric("BigEarthNet", base_encoder, "linear", "imagenet", 0.1)
        full_pre = get_metric("BigEarthNet", base_encoder, "linear", "pretrain", 1.0)
        full_imagenet = get_metric("BigEarthNet", base_encoder, "linear", "imagenet", 1.0)

        if low_pre is None or low_imagenet is None or full_pre is None or full_imagenet is None:
            status = "insufficient_data"
            details = "Need BigEarthNet linear imagenet+pretrain at train_frac 0.1 and 1.0."
        else:
            low_gain = low_pre - low_imagenet
            full_gain = full_pre - full_imagenet
            status = "pass" if low_gain > full_gain else "fail"
            details = f"gain@0.1={low_gain:.3f}, gain@1.0={full_gain:.3f}"

        claim_rows.append(
            {
                "claim_id": f"hypothesis-h1-{base_encoder}",
                "category": "hypothesis",
                "status": status,
                "dataset": "BigEarthNet",
                "base_encoder": base_encoder,
                "train_mode": "linear",
                "train_frac": "0.1_vs_1.0",
                "details": details,
            }
        )

    for dataset in datasets:
        for base_encoder in base_encoders:
            fracs_for_dataset = {1.0} if dataset == "EuroSAT" else {f for f in train_fracs if f in {0.1, 1.0}}
            for train_frac in sorted(fracs_for_dataset):
                pre_linear = get_metric(dataset, base_encoder, "linear", "pretrain", train_frac)
                random_linear = get_metric(dataset, base_encoder, "linear", "random", train_frac)
                imagenet_linear = get_metric(dataset, base_encoder, "linear", "imagenet", train_frac)
                pre_ft = get_metric(dataset, base_encoder, "finetune", "pretrain", train_frac)
                random_ft = get_metric(dataset, base_encoder, "finetune", "random", train_frac)
                imagenet_ft = get_metric(dataset, base_encoder, "finetune", "imagenet", train_frac)

                if None in {pre_linear, random_linear, imagenet_linear, pre_ft, random_ft, imagenet_ft}:
                    status = "insufficient_data"
                    details = "Need linear+finetune runs for random, imagenet, pretrain."
                else:
                    linear_gap = pre_linear - max(random_linear, imagenet_linear)
                    ft_gap = pre_ft - max(random_ft, imagenet_ft)
                    pretrain_still_best = pre_ft > max(random_ft, imagenet_ft)
                    status = "pass" if (pretrain_still_best and ft_gap <= linear_gap) else "fail"
                    details = (
                        f"linear_gap={linear_gap:.3f}, finetune_gap={ft_gap:.3f}, "
                        f"pretrain_still_best={pretrain_still_best}"
                    )

                claim_rows.append(
                    {
                        "claim_id": f"hypothesis-h2-{dataset.lower()}-{base_encoder}-frac{train_frac:g}",
                        "category": "hypothesis",
                        "status": status,
                        "dataset": dataset,
                        "base_encoder": base_encoder,
                        "train_mode": "linear_vs_finetune",
                        "train_frac": train_frac,
                        "details": details,
                    }
                )

    return claim_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results/checkin2_part2")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"results_root does not exist: {results_root}")

    runs = load_run_summaries(results_root)
    if not runs:
        raise RuntimeError(f"No summary.json files found under: {results_root}")

    aggregated = aggregate_runs(runs)
    paper_comparison = build_paper_comparison_rows(aggregated)
    claim_checks = build_claim_check_rows(aggregated)

    tables_dir = results_root / "tables"
    run_metrics_path = tables_dir / "run_metrics.csv"
    aggregated_path = tables_dir / "aggregated_metrics.csv"
    paper_path = tables_dir / "paper_vs_ours.csv"
    claims_path = tables_dir / "claim_checks.csv"

    write_csv(
        run_metrics_path,
        runs,
        [
            "dataset",
            "run_name",
            "base_encoder",
            "train_mode",
            "backbone_type",
            "seed",
            "train_frac",
            "best_metric",
            "best_epoch",
            "final_metric",
            "metrics_csv",
            "summary_path",
        ],
    )
    write_csv(
        aggregated_path,
        aggregated,
        [
            "dataset",
            "base_encoder",
            "train_mode",
            "backbone_type",
            "train_frac",
            "n_runs",
            "seeds",
            "mean_best_metric",
            "std_best_metric",
            "mean_final_metric",
            "std_final_metric",
        ],
    )
    write_csv(
        paper_path,
        paper_comparison,
        [
            "dataset",
            "base_encoder",
            "train_mode",
            "train_frac",
            "backbone_type",
            "n_runs",
            "seeds",
            "ours_mean_best",
            "ours_std_best",
            "ours_mean_final",
            "paper_caco_100k",
            "paper_caco_1m",
            "delta_vs_paper_100k",
            "delta_vs_paper_1m",
            "notes",
        ],
    )
    write_csv(
        claims_path,
        claim_checks,
        [
            "claim_id",
            "category",
            "status",
            "dataset",
            "base_encoder",
            "train_mode",
            "train_frac",
            "details",
        ],
    )

    print(f"Wrote {len(runs)} run rows to {run_metrics_path}")
    print(f"Wrote {len(aggregated)} aggregated rows to {aggregated_path}")
    print(f"Wrote {len(paper_comparison)} comparison rows to {paper_path}")
    print(f"Wrote {len(claim_checks)} claim/hypothesis rows to {claims_path}")


if __name__ == "__main__":
    main()
