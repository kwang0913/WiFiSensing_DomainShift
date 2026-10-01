"""Training-only class/domain-balanced batches, expressed as a flat index sampler."""
import math
import numpy as np
from torch.utils.data import Sampler


class ClassDomainSampler(Sampler):
    def __init__(self, labels, domains, recordings, config, seed=42):
        self.rng = np.random.default_rng(seed)
        self.recordings = np.asarray(recordings)
        labels, domains = np.asarray(labels), np.asarray(domains)
        self.classes = np.unique(labels)
        if len(self.classes) < 2 or np.any(domains < 0):
            raise ValueError("Balanced sampling requires two task classes and known training domains")
        self.groups = {
            label: {domain: np.flatnonzero((labels == label) & (domains == domain))
                    for domain in np.unique(domains[labels == label])}
            for label in self.classes
        }
        self.classes_per_batch = min(config.classes_per_batch, len(self.classes))
        self.domains_per_class = config.domains_per_class
        self.samples_per_domain = config.samples_per_domain
        self.batch_size = self.classes_per_batch * self.domains_per_class * self.samples_per_domain
        self.batches = math.ceil(len(labels) / self.batch_size)

    def __len__(self):
        return self.batches * self.batch_size

    def _samples(self, indices, count):
        # Prefer distinct recordings, then fill from unused segments. Small groups
        # are oversampled only after every available segment has been selected.
        selected = []
        while len(selected) < count:
            records = self.rng.permutation(np.unique(self.recordings[indices]))
            available = indices.copy()
            cycle = []
            for record in records:
                choices = available[self.recordings[available] == record]
                cycle.append(int(self.rng.choice(choices)))
            remaining = np.setdiff1d(available, cycle)
            cycle.extend(self.rng.permutation(remaining).tolist())
            selected.extend(cycle[:count - len(selected)])
        return selected

    def __iter__(self):
        for _ in range(self.batches):
            batch = []
            for label in self.rng.choice(self.classes, self.classes_per_batch, replace=False):
                groups = self.groups[label]
                domains = self.rng.permutation(list(groups))[:self.domains_per_class].tolist()
                if len(domains) < self.domains_per_class:
                    domains.extend(self.rng.choice(list(groups), self.domains_per_class - len(domains)).tolist())
                for domain in dict.fromkeys(domains):
                    batch.extend(self._samples(groups[domain], domains.count(domain) * self.samples_per_domain))
            yield from self.rng.permutation(batch).tolist()
