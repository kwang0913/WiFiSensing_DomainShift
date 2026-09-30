"""Fit scoring models on training embeddings. Higher scores mean less conformity."""
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

class KDEScorer:
    def __init__(self, classes, kernel="gaussian", bandwidth=1.0):
        self.classes, self.kernel, self.bandwidth = classes, kernel, bandwidth

    def fit(self, x, y):
        self.models = []
        for label in range(self.classes):
            subset = x[y == label]
            if not len(subset):
                raise ValueError("Every class requires training examples")
            self.models.append(KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(subset))
        return self

    def score(self, x):
        return -np.column_stack([model.score_samples(x) for model in self.models])

class MarginScorer:
    def __init__(self, estimator):
        self.model = estimator

    def fit(self, x, y):
        self.model.fit(x, y)
        if not np.array_equal(self.model.classes_, np.arange(len(self.model.classes_))):
            raise ValueError("Class labels must be contiguous integers starting at zero")
        return self

    def score(self, x):
        probabilities = self.model.predict_proba(x)
        if probabilities.shape[1] < 2:
            raise ValueError("Margin scoring requires at least two classes")
        result = np.empty_like(probabilities)
        for label in range(probabilities.shape[1]):
            other = np.max(np.delete(probabilities, label, axis=1), axis=1)
            result[:, label] = 0.5 - (probabilities[:, label] - other) / 2
        return result


def build_scorer(config, classes, seed=42):
    """Dispatch by configuration type; no irrelevant KDE settings for SVM/HBGB."""
    from step00_config.config import KDEConfig, SVMConfig, HBGBConfig
    if isinstance(config, KDEConfig):
        return KDEScorer(classes, kernel=config.kernel, bandwidth=config.bandwidth)
    if isinstance(config, SVMConfig):
        svm = SVC(C=config.C, kernel=config.kernel, gamma=config.gamma,
                  break_ties=True, decision_function_shape="ovr")
        # Probability calibration stays inside training; Step 06 uses a separate split.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return MarginScorer(CalibratedClassifierCV(
            svm, method="sigmoid", cv=cv, ensemble=False))
    if isinstance(config, HBGBConfig):
        return MarginScorer(HistGradientBoostingClassifier(
            max_iter=config.max_iter, learning_rate=config.learning_rate,
            max_leaf_nodes=config.max_leaf_nodes, random_state=seed))
    raise TypeError(f"Unsupported score configuration: {type(config).__name__}")
