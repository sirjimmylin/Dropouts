#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


EUROSAT_ZIP_URL = "https://research.cs.cornell.edu/caco/data/eurosat/EuroSAT_RGB.zip"
EUROSAT_SPLIT_BASE = "https://research.cs.cornell.edu/caco/data/eurosat"

BIGEARTHNET_S2_URL = "https://zenodo.org/api/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz/content"
BIGEARTHNET_SPLIT_URLS = {
    "train": "https://zenodo.org/api/records/12687186/files/train.csv.gz/content",
    "val": "https://zenodo.org/api/records/12687186/files/val.csv.gz/content",
    "test": "https://zenodo.org/api/records/12687186/files/test.csv.gz/content",
}
BIGEARTHNET_BAD_PATCH_URLS = {
    "patches_with_cloud_and_shadow.csv.gz": (
        "https://zenodo.org/api/records/12687186/files/patches_with_cloud_and_shadow.csv.gz/content"
    ),
    "patches_with_seasonal_snow.csv.gz": (
        "https://zenodo.org/api/records/12687186/files/patches_with_seasonal_snow.csv.gz/content"
    ),
}


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst} exists")
        return
    print(f"[download] {url} -> {dst}")
    with urllib.request.urlopen(url) as response, dst.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_eurosat(data_root: Path) -> Path:
    eurosat_root = data_root / "eurosat"
    zip_path = eurosat_root / "EuroSAT_RGB.zip"
    dataset_dir = eurosat_root / "EuroSAT_RGB"

    download_file(EUROSAT_ZIP_URL, zip_path)

    if not dataset_dir.exists():
        print(f"[extract] {zip_path} -> {eurosat_root}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(eurosat_root)
    else:
        print(f"[skip] {dataset_dir} already extracted")

    for split in ("train", "val", "all"):
        split_path = dataset_dir / f"{split}.txt"
        split_url = f"{EUROSAT_SPLIT_BASE}/{split}.txt"
        download_file(split_url, split_path)

    return dataset_dir


def csv_gz_first_col_to_txt(csv_gz_path: Path, txt_path: Path) -> None:
    print(f"[convert] {csv_gz_path} -> {txt_path}")
    with gzip.open(csv_gz_path, "rt", encoding="utf-8") as f, txt_path.open("w", encoding="utf-8") as out:
        reader = csv.reader(f)
        for row in reader:
            if row:
                out.write(row[0].strip() + "\n")


def ungzip(src_gz: Path, dst_path: Path) -> None:
    print(f"[convert] {src_gz} -> {dst_path}")
    with gzip.open(src_gz, "rt", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
        dst.write(src.read())


def ensure_bigearthnet_metadata(data_root: Path) -> Path:
    ben_root = data_root / "bigearthnet"
    ben_root.mkdir(parents=True, exist_ok=True)

    for split, url in BIGEARTHNET_SPLIT_URLS.items():
        split_gz = ben_root / f"{split}.csv.gz"
        split_txt = ben_root / f"{split}.txt"
        download_file(url, split_gz)
        if not split_txt.exists():
            csv_gz_first_col_to_txt(split_gz, split_txt)

    for filename, url in BIGEARTHNET_BAD_PATCH_URLS.items():
        gz_path = ben_root / filename
        csv_path = ben_root / filename.replace(".gz", "")
        download_file(url, gz_path)
        if not csv_path.exists():
            ungzip(gz_path, csv_path)

    return ben_root


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    destination = destination.resolve()
    target = (destination / member.name).resolve()
    if not str(target).startswith(str(destination)):
        raise ValueError(f"Unsafe tar member path: {member.name}")
    tar.extract(member, path=destination)


def pick_subset_ids(ben_root: Path, split: str, count: int, seed: int, bad_patches: set[str]) -> list[str]:
    if count <= 0:
        return []
    ids = [x for x in read_lines(ben_root / f"{split}.txt") if x not in bad_patches]
    rng = __import__("random").Random(seed)
    rng.shuffle(ids)
    return ids[: min(count, len(ids))]


def extract_subset_from_stream(
    ben_root: Path,
    target_ids: set[str],
) -> set[str]:
    extracted = set()
    if not target_ids:
        return extracted

    output_root = ben_root
    existing_dir = ben_root / "BigEarthNet-v1.0"
    if existing_dir.exists():
        for patch_dir in existing_dir.iterdir():
            if patch_dir.is_dir() and patch_dir.name in target_ids:
                extracted.add(patch_dir.name)
    remaining = target_ids - extracted
    if not remaining:
        print("[skip] requested subset already extracted")
        return extracted

    print(
        f"[stream-extract] extracting {len(remaining)} patches from BigEarthNet-S2 archive stream; "
        "this may take a while"
    )
    with urllib.request.urlopen(BIGEARTHNET_S2_URL) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tf:
            for member in tf:
                parts = Path(member.name).parts
                if len(parts) < 2:
                    continue
                patch_id = parts[1]
                if patch_id not in remaining:
                    continue
                safe_extract_member(tf, member, output_root)
                if member.name.endswith("_labels_metadata.json"):
                    extracted.add(patch_id)
                    remaining.discard(patch_id)
                    if not remaining:
                        break
    return extracted


def maybe_prepare_bigearthnet_subset(
    ben_root: Path,
    subset_train: int,
    subset_val: int,
    subset_test: int,
    seed: int,
    activate_subset_splits: bool,
) -> None:
    if subset_train <= 0 and subset_val <= 0 and subset_test <= 0:
        return

    bad_patches = set(read_lines(ben_root / "patches_with_cloud_and_shadow.csv"))
    bad_patches.update(read_lines(ben_root / "patches_with_seasonal_snow.csv"))

    train_ids = pick_subset_ids(ben_root, "train", subset_train, seed, bad_patches)
    val_ids = pick_subset_ids(ben_root, "val", subset_val, seed + 1, bad_patches)
    test_ids = pick_subset_ids(ben_root, "test", subset_test, seed + 2, bad_patches)
    target_ids = set(train_ids + val_ids + test_ids)
    extracted_ids = extract_subset_from_stream(ben_root, target_ids)

    train_ids = [x for x in train_ids if x in extracted_ids]
    val_ids = [x for x in val_ids if x in extracted_ids]
    test_ids = [x for x in test_ids if x in extracted_ids]

    write_lines(ben_root / "train.subset.txt", train_ids)
    write_lines(ben_root / "val.subset.txt", val_ids)
    write_lines(ben_root / "test.subset.txt", test_ids)

    if activate_subset_splits:
        for split in ("train", "val", "test"):
            full_path = ben_root / f"{split}.full.txt"
            split_path = ben_root / f"{split}.txt"
            subset_path = ben_root / f"{split}.subset.txt"
            if not full_path.exists():
                shutil.copy2(split_path, full_path)
            shutil.copy2(subset_path, split_path)
        print("[info] Activated subset split files (train.txt/val.txt/test.txt now point to subset)")


def maybe_download_bigearthnet_archive(ben_root: Path) -> None:
    archive_path = ben_root / "BigEarthNet-S2-v1.0.tar.gz"
    download_file(BIGEARTHNET_S2_URL, archive_path)
    print(f"[info] Downloaded archive to {archive_path} ({archive_path.stat().st_size / (1024 ** 3):.2f} GiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--download_eurosat", action="store_true")
    parser.add_argument("--download_bigearthnet_metadata", action="store_true")
    parser.add_argument("--download_bigearthnet_archive", action="store_true")
    parser.add_argument("--subset_train", type=int, default=0)
    parser.add_argument("--subset_val", type=int, default=0)
    parser.add_argument("--subset_test", type=int, default=0)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--activate_subset_splits", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.download_eurosat:
        eurosat_dir = ensure_eurosat(data_root)
        print(f"[ready] EuroSAT directory: {eurosat_dir}")

    ben_root = None
    if args.download_bigearthnet_metadata or args.download_bigearthnet_archive or (
        args.subset_train > 0 or args.subset_val > 0 or args.subset_test > 0
    ):
        ben_root = ensure_bigearthnet_metadata(data_root)
        print(f"[ready] BigEarthNet metadata directory: {ben_root}")

    if args.download_bigearthnet_archive:
        if ben_root is None:
            ben_root = ensure_bigearthnet_metadata(data_root)
        maybe_download_bigearthnet_archive(ben_root)

    if args.subset_train > 0 or args.subset_val > 0 or args.subset_test > 0:
        if ben_root is None:
            ben_root = ensure_bigearthnet_metadata(data_root)
        maybe_prepare_bigearthnet_subset(
            ben_root=ben_root,
            subset_train=args.subset_train,
            subset_val=args.subset_val,
            subset_test=args.subset_test,
            seed=args.subset_seed,
            activate_subset_splits=args.activate_subset_splits,
        )


if __name__ == "__main__":
    main()
