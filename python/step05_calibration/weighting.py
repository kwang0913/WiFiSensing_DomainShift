"""Estimate an target/source density ratio on frozen, clean embeddings."""
import numpy as np
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def weight_summary(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Expected nonempty finite positive weights")
    scaled = weights / weights.max()
    return {"samples": len(weights), "ess": float(scaled.sum() ** 2 / (scaled @ scaled)),
            "quantiles": dict(zip(("min", "p05", "median", "p95", "max"),
                                  np.quantile(weights, [0, .05, .5, .95, 1]).tolist()))}


def source_matching_weights(source_strata, calibration_strata):
    """Match source domain/class proportions to calibration, without resampling.

    Only source/calibration labels enter this calculation; target activity
    labels are never used. Does not correct within-stratum distribution shift.
    """
    from collections import Counter
    source, calibration = list(map(tuple, source_strata)), list(map(tuple, calibration_strata))
    ns, nc = Counter(source), Counter(calibration)
    if not source or not calibration or set(nc) - set(ns):
        raise ValueError("Every calibration domain/class stratum needs weight-training source samples")
    return np.array([nc[s] / len(calibration) / (ns[s] / len(source)) for s in source])


class DensityRatioEstimator:
    def __init__(self, config, seed=42):
        # Later notebook configuration edits must not change a fitted ratio.
        self.config = deepcopy(config)
        self.seed = seed

    def fit(self, source, target, source_weights=None):
        source, target = np.asarray(source), np.asarray(target)
        if (source.ndim != 2 or target.ndim != 2 or not len(source) or not len(target)
                or source.shape[1] != target.shape[1]
                or not np.isfinite(source).all() or not np.isfinite(target).all()):
            raise ValueError("Expected nonempty finite source/target embedding matrices")
        source_weights = np.ones(len(source)) if source_weights is None else np.asarray(source_weights, dtype=float)
        if (source_weights.shape != (len(source),) or not np.isfinite(source_weights).all()
                or (source_weights < 0).any() or source_weights.sum() <= 0):
            raise ValueError("Invalid source matching weights")
        # Preserve the source prior while matching calibration stratum proportions.
        source_weights = source_weights / source_weights.sum() * len(source)
        x = np.concatenate([source, target])
        y = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
        sample_weights = np.r_[source_weights, np.ones(len(target))]
        self.scaler = StandardScaler().fit(x, sample_weight=sample_weights)
        self.model = LogisticRegression(C=self.config.C, max_iter=self.config.max_iter,
                                        random_state=self.seed)
        self.model.fit(self.scaler.transform(x), y, sample_weight=sample_weights)
        self.log_prior_correction = np.log(len(source) / len(target))
        self.fit_summary = {"source_samples": len(source), "target_samples": len(target),
                            "source_matching_ess": weight_summary(source_weights[source_weights > 0])["ess"],
                            "iterations": self.model.n_iter_.tolist(),
                            "converged": bool(np.max(self.model.n_iter_) < self.config.max_iter)}
        return self

    def weights(self, embeddings):
        x = np.asarray(embeddings)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Expected a finite embedding matrix")
        log_ratio = self.model.decision_function(self.scaler.transform(x)) + self.log_prior_correction
        if self.config.clip_min is not None:
            log_ratio = np.maximum(log_ratio, np.log(self.config.clip_min))
        if self.config.clip_max is not None:
            log_ratio = np.minimum(log_ratio, np.log(self.config.clip_max))
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp(log_ratio)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Density ratio overflow/underflow; review domain overlap or explicit clipping settings")
        return weights


class DomainMixtureEstimator:
    """Fit target mixture proportions over source domains using unlabeled target features.

    Source-domain classifier posteriors approximate component density ratios.
    This is a plug-in mixture model, not a guarantee of target support/coverage.
    """
    def __init__(self, config, seed=42):
        self.config = deepcopy(config)
        self.seed = seed

    def fit(self, source, target, source_weights=None, source_domains=None):
        source, target = np.asarray(source), np.asarray(target)
        domains = np.asarray(source_domains)
        if (source.ndim != 2 or target.ndim != 2 or not len(source) or not len(target)
                or source.shape[1] != target.shape[1] or not np.isfinite(source).all()
                or not np.isfinite(target).all() or domains.shape != (len(source),)):
            raise ValueError("Expected finite features and one source domain per row")
        sw = np.ones(len(source)) if source_weights is None else np.asarray(source_weights, dtype=float)
        if sw.shape != (len(source),) or not np.isfinite(sw).all() or (sw < 0).any() or sw.sum() <= 0:
            raise ValueError("Invalid source weights")
        keep = sw > 0
        source, domains, sw = source[keep], domains[keep], sw[keep]
        if len(np.unique(domains)) < 2:
            raise ValueError("Domain mixture requires at least two supported source domains")
        sw = sw / sw.sum() * len(sw)
        self.scaler = StandardScaler().fit(source, sample_weight=sw)
        self.model = LogisticRegression(C=self.config.C, max_iter=self.config.max_iter, random_state=self.seed)
        self.model.fit(self.scaler.transform(source), domains, sample_weight=sw)
        self.priors = np.array([sw[domains == d].sum() for d in self.model.classes_]) / sw.sum()
        component_ratios = self.model.predict_proba(self.scaler.transform(target)) / self.priors
        mixture = self.priors.copy()
        converged = False
        for iteration in range(self.config.max_iter):
            responsibilities = component_ratios * mixture
            responsibilities /= responsibilities.sum(1, keepdims=True)
            updated = responsibilities.mean(0)
            if np.max(np.abs(updated - mixture)) < 1e-8:
                mixture, converged = updated, True
                break
            mixture = updated
        self.mixture = mixture
        self.fit_summary = {"source_samples": len(source), "target_samples": len(target),
                            "source_matching_ess": weight_summary(sw)["ess"],
                            "domains": self.model.classes_.tolist(), "source_priors": self.priors.tolist(),
                            "target_mixture": mixture.tolist(), "iterations": iteration + 1,
                            "converged": converged,
                            "interpretation": "Estimated source-domain mixture; target may lie outside mixture family."}
        return self

    def weights(self, embeddings):
        x = np.asarray(embeddings)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Expected finite embeddings")
        weights = self.model.predict_proba(self.scaler.transform(x)) @ (self.mixture / self.priors)
        if self.config.clip_min is not None:
            weights = np.maximum(weights, self.config.clip_min)
        if self.config.clip_max is not None:
            weights = np.minimum(weights, self.config.clip_max)
        weight_summary(weights)
        return weights
