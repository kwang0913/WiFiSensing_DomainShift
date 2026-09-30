"""Training-only noise and masks for the stored CSI feature layout."""
import math

import torch


class BatchAugmenter:
    """Own an RNG independent of shuffle/Dropout; advance it on every batch."""

    def __init__(self, config, seed, device):
        self.config = config
        self.generator = torch.Generator(device=device).manual_seed(seed)
        noise, mask = config.time_noise, config.stft_mask
        time_mask = config.time_mask
        if noise["kind"] not in ("none", "gaussian", "uniform"):
            raise ValueError(f"Unknown noise kind: {noise['kind']}")
        if not math.isfinite(noise["relative_std"]) or noise["relative_std"] < 0:
            raise ValueError("Noise relative_std must be finite and nonnegative")
        for value in (noise["probability"], mask["probability"],
                      mask["time_max_fraction"], mask["frequency_max_fraction"],
                      time_mask["probability"], time_mask["max_fraction"]):
            if not 0 <= value <= 1:
                raise ValueError("Augmentation probabilities and fractions must be in [0, 1]")
        if not isinstance(mask["frequency_bins"], int) or mask["frequency_bins"] < 1:
            raise ValueError("STFT frequency_bins must be a positive integer")

    def _uniform(self, shape, x):
        return torch.rand(shape, device=x.device, dtype=x.dtype, generator=self.generator)

    def _axis_mask(self, length, fraction, probability, x):
        # One contiguous band per sample, shared by all features/channels.
        # floor keeps the configured maximum fraction a strict upper bound.
        maximum = int(length * fraction)
        positions = torch.arange(length, device=x.device)[None, :]
        widths = torch.randint(maximum + 1, (len(x), 1), device=x.device,
                               generator=self.generator)
        starts = (self._uniform((len(x), 1), x) * (length - widths + 1)).long()
        selected = self._uniform((len(x), 1), x) < probability
        return ~((positions >= starts) & (positions < starts + widths) & selected)

    @torch.no_grad()
    def __call__(self, time, stft):
        if not self.config.enabled:
            return time, stft
        noise, mask = self.config.time_noise, self.config.stft_mask
        if (noise["enabled"] and noise["kind"] != "none"
                and noise["relative_std"] > 0 and noise["probability"] > 0):
            if noise["kind"] == "gaussian":
                perturbation = torch.randn(time.shape, device=time.device, dtype=time.dtype,
                                           generator=self.generator)
            else:
                perturbation = (self._uniform(time.shape, time) * 2 - 1) * math.sqrt(3)
            scale = time.square().mean(dim=(-2, -1), keepdim=True).sqrt()
            selected = self._uniform((len(time), 1, 1, 1), time) < noise["probability"]
            time = time + perturbation * scale * noise["relative_std"] * selected
        time_mask = self.config.time_mask
        if time_mask["enabled"] and time_mask["probability"] > 0 and time_mask["max_fraction"] > 0:
            # Mask after noise so hidden time intervals stay exactly zero.
            keep = self._axis_mask(time.shape[-1], time_mask["max_fraction"],
                                   time_mask["probability"], time)
            time = time * keep[:, None, None, :]
        if (mask["enabled"] and mask["probability"] > 0
                and (mask["time_max_fraction"] > 0 or mask["frequency_max_fraction"] > 0)):
            bins = mask["frequency_bins"]
            if stft.shape[-1] % bins:
                raise ValueError("STFT width must be divisible by frequency_bins")
            frames = stft.shape[-1] // bins
            # MATLAB piece(:) stores frequency bins fastest, then time frames.
            spectrum = stft.reshape(*stft.shape[:-1], frames, bins)
            time_keep = self._axis_mask(frames, mask["time_max_fraction"], mask["probability"], stft)
            freq_keep = self._axis_mask(bins, mask["frequency_max_fraction"], mask["probability"], stft)
            keep = time_keep[:, :, None] & freq_keep[:, None, :]
            stft = (spectrum * keep[:, None, None]).reshape_as(stft)
        return time, stft
