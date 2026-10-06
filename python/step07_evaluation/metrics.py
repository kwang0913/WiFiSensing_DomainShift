"""Summarize regular CP and APS predictions without model or filesystem access."""
import numpy as np
from step06_prediction.predict import prediction_sets
from .evaluate import evaluate


def summarize_predictions(labels, test_sets, point_predictions, nn_predictions,
                          method_p_values, alpha_grid, alpha, class_names, experiment, *,
                          primary_method):
    nn_accuracy = float((nn_predictions == labels).mean())
    metrics = {**evaluate(labels, test_sets, point_predictions),
                    "cnn_classification_accuracy": nn_accuracy,
                    "alpha": alpha, "nominal_coverage": 1 - alpha}
    per_class = {name: evaluate(labels[labels == i], test_sets[labels == i], point_predictions[labels == i])
                      if (labels == i).any() else {"samples": 0} for i, name in enumerate(class_names)}
    method_results = {}
    for name, values in method_p_values.items():
        point = point_predictions if name == primary_method else nn_predictions
        method_results[name] = [{"alpha": level, "nominal_coverage": 1 - level,
                                     **evaluate(labels, prediction_sets(values, level), point)}
                                    for level in alpha_grid]
    alpha_results = method_results[primary_method]
    per_domain = {}
    if experiment["split"]["mode"] == "domain_holdout":
        key = experiment["data"]["domain_key"]
        domains = np.asarray([experiment["recordings"][i]["metadata"][key]
                              for i, _ in experiment["splits"]["test"]])
        for domain in sorted(set(domains)):
            mask = domains == domain
            per_domain[domain] = evaluate(labels[mask], test_sets[mask], point_predictions[mask])
    return dict(nn_accuracy=nn_accuracy, metrics=metrics, per_class=per_class,
                method_results=method_results, alpha_results=alpha_results, per_domain=per_domain)
