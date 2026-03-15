import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import numpy as np
from os import PathLike

ORIG_H, ORIG_W = 180, 320
PAD_H = 192

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def load_npy(path, mmap_mode="r"):
    return np.load(path, mmap_mode=mmap_mode)


def load_npy_pair(images_path, labels_path, mmap_mode="r"):
    images = load_npy(images_path, mmap_mode=mmap_mode)
    labels = load_npy(labels_path, mmap_mode=mmap_mode)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same number of samples")
    return images, labels

def pad_to_192(img_chw: torch.Tensor, mask_hw: torch.Tensor):
    pad_bottom = PAD_H - ORIG_H
    img = F.pad(img_chw, (0, 0, 0, pad_bottom), mode="constant", value=0.0)
    mask = F.pad(mask_hw, (0, 0, 0, pad_bottom), mode="constant", value=-1)
    return img, mask

class BDDDataset(Dataset):
    def __init__(self, images, labels, mmap_mode=None):
        if isinstance(images, (str, PathLike)):
            images = load_npy(images, mmap_mode=mmap_mode)
        if isinstance(labels, (str, PathLike)):
            labels = load_npy(labels, mmap_mode=mmap_mode)

        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same number of samples")

        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img_np = self.images[idx]
        mask_np = self.labels[idx]

        if not img_np.flags.writeable:
            img_np = img_np.copy()
        if not mask_np.flags.writeable:
            mask_np = mask_np.copy()

        img = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask_np).long()
        img, mask = pad_to_192(img, mask)
        img = (img - MEAN) / STD
        return img, mask

def split_data(full_ds, train_ratio=0.8, val_ratio=None, train_size=None, seed=42):
    N = len(full_ds)

    if train_size is not None:
        n_train = train_size
    else:
        n_train = int(N * train_ratio)

    if n_train > N:
        raise ValueError("train_size exceeds dataset size")

    if val_ratio is not None:
        n_val = int(N * val_ratio)
        if n_train + n_val > N:
            raise ValueError("train + val exceeds dataset size")

        n_test = N - n_train - n_val
        lengths = [n_train, n_val, n_test]
    else:
        n_val = N - n_train
        lengths = [n_train, n_val]

    generator = torch.Generator().manual_seed(seed)

    splits = random_split(full_ds, lengths, generator=generator)

    return splits