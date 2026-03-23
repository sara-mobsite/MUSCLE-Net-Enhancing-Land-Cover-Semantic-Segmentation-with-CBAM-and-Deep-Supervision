import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class Sentinel2Dataset(Dataset):
    def __init__(self, sentinel1_dir: str, sentinel2_dir: str, label_dir: str):
        self.sentinel1_dir = sentinel1_dir
        self.sentinel2_dir = sentinel2_dir
        self.label_dir = label_dir
        self.data_pairs = self._load_data_pairs()

        self.band_indices_sentinel1 = [1, 2]
        self.band_indices_sentinel2 = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]

        self.class_map = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7}

        self.mean_vals = [
            -12.64386368, -19.35255814,
            438.37207031, 614.05566406, 588.40960693, 942.84332275, 1769.93164062,
            2049.55151367, 2193.29199219, 2235.55664062, 1568.22680664, 997.7324829
        ]
        self.std_vals = [
            5.1334939, 5.5905056,
            607.02685547, 603.29681396, 684.56884766, 738.43267822, 1100.45605469,
            1275.80541992, 1369.3717041, 1356.54406738, 1070.16125488, 813.52764893
        ]

    def _load_data_pairs(self):
        data_pairs = []
        for file in sorted(os.listdir(self.sentinel1_dir)):
            if not file.endswith(".tif"):
                continue

            sentinel1_path = os.path.join(self.sentinel1_dir, file)
            label_file_name = file.replace("s1", "dfc")
            label_path = os.path.join(self.label_dir, label_file_name)
            s2_file_name = file.replace("s1", "s2")
            sentinel2_path = os.path.join(self.sentinel2_dir, s2_file_name)

            if os.path.exists(sentinel2_path) and os.path.exists(label_path):
                data_pairs.append((sentinel1_path, sentinel2_path, label_path))

        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        sentinel1_path, sentinel2_path, label_path = self.data_pairs[idx]

        with rasterio.open(sentinel1_path) as src1:
            sentinel1 = src1.read(self.band_indices_sentinel1)

        with rasterio.open(sentinel2_path) as src2:
            sentinel2 = src2.read(self.band_indices_sentinel2)

        combined = torch.tensor(
            np.concatenate([sentinel1, sentinel2], axis=0),
            dtype=torch.float32
        )

        for i in range(combined.shape[0]):
            combined[i] = (combined[i] - self.mean_vals[i]) / self.std_vals[i]

        with rasterio.open(label_path) as src:
            label_image = src.read(1).astype("int64")

        label_image = torch.tensor(label_image, dtype=torch.long)
        for original, new in self.class_map.items():
            label_image[label_image == original] = new

        return combined, label_image
