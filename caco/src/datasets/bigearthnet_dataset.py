import csv
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, download_url

try:
    import rasterio
except ImportError:  # pragma: no cover - fallback for environments without GDAL wheels
    rasterio = None
    import tifffile

ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
RGB_BANDS = ['B04', 'B03', 'B02']

BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}

LABELS = [
    'Agro-forestry areas', 'Airports',
    'Annual crops associated with permanent crops', 'Bare rock',
    'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
    'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
    'Construction sites', 'Continuous urban fabric',
    'Discontinuous urban fabric', 'Dump sites', 'Estuaries',
    'Fruit trees and berry plantations', 'Green urban areas',
    'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
    'Land principally occupied by agriculture, with significant areas of '
    'natural vegetation', 'Mineral extraction sites', 'Mixed forest',
    'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
    'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land',
    'Port areas', 'Rice fields', 'Road and rail networks and associated land',
    'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
    'Sparsely vegetated areas', 'Sport and leisure facilities',
    'Transitional woodland/shrub', 'Vineyards', 'Water bodies', 'Water courses'
]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class Bigearthnet(Dataset):
    # Original URL from SeCo/CACo no longer resolves; use official Zenodo v1.0 record.
    url = 'https://zenodo.org/api/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz/content'
    archive_name = 'BigEarthNet-S2-v1.0.tar.gz'
    subdir = 'BigEarthNet-v1.0'
    list_file = {
        'train': 'https://zenodo.org/api/records/12687186/files/train.csv.gz/content',
        'val': 'https://zenodo.org/api/records/12687186/files/val.csv.gz/content',
        'test': 'https://zenodo.org/api/records/12687186/files/test.csv.gz/content'
    }
    bad_patches = [
        'https://zenodo.org/api/records/12687186/files/patches_with_seasonal_snow.csv.gz/content',
        'https://zenodo.org/api/records/12687186/files/patches_with_cloud_and_shadow.csv.gz/content'
    ]

    def __init__(self, root, split, bands=None, transform=None, target_transform=None, download=False, use_new_labels=True):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else RGB_BANDS
        self.transform = transform
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels

        if download:
            download_and_extract_archive(self.url, self.root, filename=self.archive_name)
            for split, split_url in self.list_file.items():
                download_url(split_url, self.root, f'{split}.csv.gz')
            for url in self.bad_patches:
                filename = Path(url).name
                if filename == 'content':
                    # Keep deterministic names even though Zenodo "content" endpoint has no filename.
                    if 'seasonal_snow' in url:
                        filename = 'patches_with_seasonal_snow.csv.gz'
                    else:
                        filename = 'patches_with_cloud_and_shadow.csv.gz'
                download_url(url, self.root, filename=filename)

        self._ensure_split_text_file(self.split)
        self._ensure_bad_patch_file('patches_with_seasonal_snow')
        self._ensure_bad_patch_file('patches_with_cloud_and_shadow')

        subdir = self._resolve_subdir()

        bad_patches = set()
        for filename in ['patches_with_seasonal_snow.csv', 'patches_with_cloud_and_shadow.csv']:
            file_path = self.root / filename
            if file_path.exists():
                with file_path.open() as f:
                    bad_patches.update(f.read().splitlines())

        self.samples = []
        skipped_missing = 0
        with open(self.root / f'{self.split}.txt') as f:
            for patch_id in f.read().splitlines():
                if patch_id in bad_patches:
                    continue

                patch_dir = self.root / subdir / patch_id
                if not self._is_complete_patch(patch_dir, patch_id):
                    skipped_missing += 1
                    continue
                self.samples.append(patch_dir)

        if skipped_missing:
            print(f"[BigEarthNet:{self.split}] skipped {skipped_missing} missing/incomplete patches from split list")
        if not self.samples:
            raise RuntimeError(
                f"No valid BigEarthNet samples found for split '{self.split}' under {self.root / subdir}. "
                "Check split files and extracted patch contents."
            )

    def _resolve_subdir(self):
        candidates = ['BigEarthNet-v1.0', 'BigEarthNet-S2-v1.0']
        for candidate in candidates:
            if (self.root / candidate).exists():
                return candidate
        return self.subdir

    def _ensure_bad_patch_file(self, stem):
        csv_path = self.root / f'{stem}.csv'
        if csv_path.exists():
            return
        gz_path = self.root / f'{stem}.csv.gz'
        if not gz_path.exists():
            return
        with gzip.open(gz_path, 'rt', encoding='utf-8') as src, csv_path.open('w', encoding='utf-8') as dst:
            dst.write(src.read())

    def _ensure_split_text_file(self, split):
        txt_path = self.root / f'{split}.txt'
        if txt_path.exists():
            return

        csv_gz_path = self.root / f'{split}.csv.gz'
        csv_path = self.root / f'{split}.csv'

        if csv_gz_path.exists():
            reader = csv.reader(gzip.open(csv_gz_path, 'rt', encoding='utf-8'))
        elif csv_path.exists():
            reader = csv.reader(csv_path.open('r', encoding='utf-8'))
        else:
            return

        with txt_path.open('w', encoding='utf-8') as out:
            for row in reader:
                if row:
                    out.write(row[0].strip() + '\n')

    def _is_complete_patch(self, patch_dir, patch_id):
        if not patch_dir.is_dir():
            return False

        label_path = patch_dir / f'{patch_id}_labels_metadata.json'
        if not label_path.exists():
            return False

        for band in self.bands:
            tif_path = patch_dir / f'{patch_id}_{band}.tif'
            if not tif_path.exists():
                return False
        return True

    def __getitem__(self, index):
        path = self.samples[index]
        patch_id = path.name

        channels = []
        for b in self.bands:
            tif_path = path / f'{patch_id}_{b}.tif'
            if rasterio is not None:
                ch = rasterio.open(tif_path).read(1)
            else:
                ch = tifffile.imread(tif_path)
            ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
            channels.append(ch)
        img = np.dstack(channels)
        img = Image.fromarray(img)

        with open(path / f'{patch_id}_labels_metadata.json', 'r') as f:
            labels = json.load(f)['labels']
        if self.use_new_labels:
            target = self.get_multihot_new(labels)
        else:
            target = self.get_multihot_old(labels)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_multihot_old(labels):
        target = np.zeros((len(LABELS),), dtype=np.float32)
        for label in labels:
            target[LABELS.index(label)] = 1
        return target

    @staticmethod
    def get_multihot_new(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            elif label not in set(NEW_LABELS):
                continue
            else:
                target[NEW_LABELS.index(label)] = 1
        return target


if __name__ == '__main__':
    import os
    import argparse
    from utils.data import make_lmdb

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    train_dataset = Bigearthnet(
        root=args.data_dir,
        split='train'
    )
    make_lmdb(train_dataset, lmdb_file=os.path.join(args.save_dir, 'train.lmdb'))

    val_dataset = Bigearthnet(
        root=args.data_dir,
        split='val'
    )
    make_lmdb(val_dataset, lmdb_file=os.path.join(args.save_dir, 'val.lmdb'))
