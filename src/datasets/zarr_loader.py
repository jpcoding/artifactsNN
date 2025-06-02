import zarr
import torch
from torch.utils.data import Dataset
import numpy as np

class ZarrCompressionDataset(Dataset):
    def __init__(self, zarr_root, sample_index, patch_size=(64, 64, 64), use_original=True):
        self.root = zarr.open(zarr_root, mode='r')
        self.samples = sample_index
        self.patch_size = patch_size
        self.use_original = use_original

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        field = sample["field"]
        timestep = sample["timestep"]
        comp = sample["compressor"]
        eb = sample["eb"]
        z, y, x = sample["z"], sample["y"], sample["x"]
        dz, dy, dx = self.patch_size

        # Decompressed input
        decompressed = self.root[f'decompressed/{timestep}/{field}/{comp}/{eb}']
        input_patch = decompressed[z:z+dz, y:y+dy, x:x+dx]

        # Target (original data)
        if self.use_original:
            original = self.root[f'original/{timestep}/{field}']
            target_patch = original[z:z+dz, y:y+dy, x:x+dx]
            return (
                torch.from_numpy(input_patch).float().unsqueeze(0),
                torch.from_numpy(target_patch).float().unsqueeze(0)
            )
        else:
            return torch.from_numpy(input_patch).float().unsqueeze(0)
