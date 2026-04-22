"""UC Merced Land Use dataset (21 classes, 2,100 images, 256x256 aerial).

Mirrors the EurosatDataset API/contract so the same linear-probing pipeline
can be reused. Class is inferred from the parent directory of each image.
"""
import re
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image

# UCM filenames look like "agricultural00.tif", "baseballdiamond42.tif".
# Class = leading alphabetic chars (strip trailing digits + extension).
_STEM_RE = re.compile(r'^([a-zA-Z]+)\d+$')


class UCMercedDataset(Dataset):
    def __init__(self, root, split, transform=None, get_path=False):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.get_path = get_path

        with open(self.root / f'{split}.txt') as f:
            filenames = [ln.strip() for ln in f if ln.strip()]

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for fn in filenames:
            stem = Path(fn).stem
            m = _STEM_RE.match(stem)
            if m is None:
                raise ValueError(f'Cannot parse class from UCM filename {fn!r}')
            cls_name = m.group(1)
            self.samples.append(self.root / cls_name / fn)

    def __getitem__(self, index):
        path = self.samples[index]
        img = Image.open(path).convert('RGB')
        target = self.class_to_idx[path.parts[-2]]
        if self.transform is not None:
            img = self.transform(img)
        if self.get_path:
            return img, target, str(path)
        return img, target

    def __len__(self):
        return len(self.samples)
