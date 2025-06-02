import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BinaryNumpyDataset(Dataset):
    def __init__(self, file_list, shape, dtype=np.float32, transform=None):
        """
        Args:
            file_list: list of paths to binary files
            shape: tuple like (C, H, W) or (H, W)
            dtype: np.float32 or np.float64 (must match what you saved)
            transform: optional transform (e.g., normalize, augment)
        """
        self.file_list = file_list
        self.shape = shape
        self.dtype = dtype
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        array = np.fromfile(file_path, dtype=self.dtype).reshape(self.shape)

        if len(self.shape) == 2:
            array = array[np.newaxis, :, :]  # add channel dimension

        tensor = torch.from_numpy(array)
        if self.transform:
            tensor = self.transform(tensor)

        return tensor


def compute_data_mean(file_list, shape, dtype=np.float32):
    means = []
    for fname in file_list:
        arr = np.fromfile(fname, dtype=dtype).reshape(shape)
        means.append(np.mean(arr))
    return np.mean(means)


class PairedBinaryNumpyDataset(Dataset):
    def __init__(self, input_file_list, target_file_list, shape, dtype=np.float32, data_mean=0.0, transform=None, is_residual=False):
        assert len(input_file_list) == len(target_file_list)
        self.input_file_list = input_file_list
        self.target_file_list = target_file_list
        self.shape = shape
        self.dtype = dtype
        self.data_mean = data_mean
        self.transform = transform
        self.is_residual = is_residual

    def set_data_mean(self, mean):
        self.data_mean = mean

    def __len__(self):
        return len(self.input_file_list)

    def __getitem__(self, idx):
        input_array = np.fromfile(self.input_file_list[idx], dtype=self.dtype).reshape(self.shape)
        target_array = np.fromfile(self.target_file_list[idx], dtype=self.dtype).reshape(self.shape)

        if len(self.shape) == 2:
            input_array = input_array[np.newaxis, :, :]
            target_array = target_array[np.newaxis, :, :]
        elif len(self.shape) == 3:
            input_array = input_array[np.newaxis, :, :, :]  # [1, D, H, W]
            target_array = target_array[np.newaxis, :, :, :]

        input_array -= self.data_mean
        if self.is_residual:
            target_array -= input_array

        input_tensor = torch.from_numpy(input_array)
        target_tensor = torch.from_numpy(target_array)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor


class PairedNumpyDatasetFromMem(Dataset):
    def __init__(self, input_array_list, target_array_list, data_mean=0.0, transform=None, is_residual=False):
        assert len(input_array_list) == len(target_array_list)
        self.input_array_list = input_array_list
        self.target_array_list = target_array_list
        self.data_mean = data_mean
        self.transform = transform
        self.is_residual = is_residual

    def __len__(self):
        return len(self.input_array_list)

    def __getitem__(self, idx):
        input_array = self.input_array_list[idx].astype(np.float32)
        target_array = self.target_array_list[idx].astype(np.float32)

        if input_array.ndim == 2:
            input_array = input_array[np.newaxis, :, :]
            target_array = target_array[np.newaxis, :, :]
        elif input_array.ndim == 3:
            input_array = input_array[np.newaxis, :, :, :]
            target_array = target_array[np.newaxis, :, :, :]

        input_array -= self.data_mean
        if self.is_residual:
            target_array -= input_array

        input_tensor = torch.from_numpy(input_array)
        target_tensor = torch.from_numpy(target_array)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor


class PairedBinaryNumpyDatasetResidual(Dataset):
    def __init__(self, input_file_list, target_file_list, shape, dtype=np.float32, transform=None):
        assert len(input_file_list) == len(target_file_list)
        self.input_file_list = input_file_list
        self.target_file_list = target_file_list
        self.shape = shape
        self.dtype = dtype
        self.transform = transform

    def __len__(self):
        return len(self.input_file_list)

    def __getitem__(self, idx):
        # Load raw arrays
        input_array = np.fromfile(self.input_file_list[idx], dtype=self.dtype).reshape(self.shape)
        target_array = np.fromfile(self.target_file_list[idx], dtype=self.dtype).reshape(self.shape)

        # Add channel dimension if needed
        if len(self.shape) == 2:
            input_array = input_array[np.newaxis, :, :]
            target_array = target_array[np.newaxis, :, :]

        # Compute residual
        residual_array = target_array - input_array

        # Convert to tensors
        input_tensor = torch.from_numpy(input_array)
        residual_tensor = torch.from_numpy(residual_array)

        # Apply any transforms
        if self.transform:
            input_tensor = self.transform(input_tensor)
            residual_tensor = self.transform(residual_tensor)

        return input_tensor, residual_tensor


class PairedBinaryNumpyDatasetResidualE(Dataset):
    def __init__(self, input_file_list, target_file_list, shape, dtype=np.float32, transform=None):
        assert len(input_file_list) == len(target_file_list)
        self.input_file_list = input_file_list
        self.target_file_list = target_file_list
        self.shape = shape
        self.dtype = dtype
        self.transform = transform

    def __len__(self):
        return len(self.input_file_list)

    def __getitem__(self, idx):
        input_array = np.fromfile(self.input_file_list[idx], dtype=self.dtype).reshape(self.shape)
        target_array = np.fromfile(self.target_file_list[idx], dtype=self.dtype).reshape(self.shape)

        if len(self.shape) == 2:
            input_array = input_array[np.newaxis, :, :]
            target_array = target_array[np.newaxis, :, :]

        residual_array = target_array - input_array
        e_scalar = np.abs(residual_array).max()

        input_tensor = torch.from_numpy(input_array).float()
        residual_tensor = torch.from_numpy(residual_array).float()
        e_tensor = torch.tensor(e_scalar, dtype=torch.float32)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            residual_tensor = self.transform(residual_tensor)

        return input_tensor, residual_tensor, e_tensor


def compute_block_coords(shape, block_size=64, stride=32):
    coords = []
    for x in range(0, shape[0], stride):
        for y in range(0, shape[1], stride):
            for z in range(0, shape[2], stride):
                x2 =int(min(x + block_size, shape[0]))
                y2 = int(min(y + block_size, shape[1]))
                z2 = int(min(z + block_size, shape[2]))

                if x2 - x != block_size or y2 - y != block_size or z2 - z != block_size:
                    continue  # skip partial blocks at boundary

                coords.append((x, y, z))
    return coords



class PairedBlockDataset(Dataset):
    def __init__(self, orig_files, quant_files, shape, block_size=64, stride=32, dtype=np.float32):
        self.orig_files = orig_files
        self.quant_files = quant_files
        self.shape = shape
        self.block_size = block_size
        self.stride = stride
        self.dtype = dtype


        # Precompute all block coordinates for all samples
        self.block_coords = []
        for file_idx in range(len(orig_files)):
            coords = coords = compute_block_coords(self.shape, self.block_size, self.stride)
            self.block_coords.extend([(file_idx, x, y, z) for (x, y, z) in coords])

    def __len__(self):
        return len(self.block_coords)

    def __getitem__(self, idx):
        file_idx, x, y, z = self.block_coords[idx]
        orig = np.fromfile(self.orig_files[file_idx], dtype=self.dtype).reshape(self.shape)
        quant = np.fromfile(self.quant_files[file_idx], dtype=self.dtype).reshape(self.shape)
        orig_block = orig[x:x+64, y:y+64, z:z+64]
        quant_block = quant[x:x+64, y:y+64, z:z+64]
        return torch.from_numpy(quant_block.copy()).unsqueeze(0), torch.from_numpy(orig_block.copy()).unsqueeze(0)