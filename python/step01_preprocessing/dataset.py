"""Memory-mapped paired segments, shared by training and evaluation."""
import numpy as np
import torch
from torch.utils.data import Dataset
from .standardize import standardize_time


class NpySegments(Dataset):
    """Read one segment at a time; keep only the latest recording pair mapped."""
    def __init__(self, time_files, stft_files, labels, samples, domain_labels=None):
        self.time_files = time_files
        self.stft_files = stft_files
        self.labels = labels
        self.samples = samples
        self.domain_labels = domain_labels
        self.current_file = None
        self.time = self.stft = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_index, segment_index = self.samples[index]
        if file_index != self.current_file:
            self.time = self.stft = None
            self.time = np.load(self.time_files[file_index], mmap_mode="r")
            self.stft = np.load(self.stft_files[file_index], mmap_mode="r")
            self.current_file = file_index
        # Copy only this segment: tensors must not write into read-only NPY mappings.
        time = np.array(self.time[segment_index], copy=True)
        stft = np.array(self.stft[segment_index], copy=True)
        if not np.isfinite(time).all() or not np.isfinite(stft).all():
            raise ValueError(f"Nonfinite features: {self.time_files[file_index]}")
        time = standardize_time(time[None])[0]
        sample = (torch.from_numpy(time).permute(2, 0, 1),
                torch.from_numpy(stft).permute(2, 0, 1),
                int(self.labels[file_index]))
        return sample if self.domain_labels is None else (*sample, int(self.domain_labels[file_index]))
