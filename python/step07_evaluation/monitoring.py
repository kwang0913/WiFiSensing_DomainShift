"""Optional evaluation logging; local results remain authoritative."""
import warnings


def log_evaluation(evaluation, monitor_config):
    if not monitor_config["enabled"]:
        return
    try:
        import wandb

        config = {key: value for key, value in evaluation.cp_config.items()
                  if key not in ("checkpoint", "split_manifest")}
        run = wandb.init(
            project=monitor_config["project"], entity=monitor_config.get("entity"),
            mode=monitor_config["mode"], dir=str(evaluation.cp_dir),
            settings={"quiet": True}, name=evaluation.cp_dir.name,
            group=evaluation.run_dir.name, job_type="conformal", config=config,
        )
        with run:
            summary = {f"test/{key}": value for key, value in evaluation.metrics.items()
                       if value is not None}
            run.summary.update(summary)
            run.summary["score_validation_accuracy"] = evaluation.score_validation_accuracy
            run.define_metric("alpha")
            run.define_metric("sweep/*", step_metric="alpha")
            for name, rows in evaluation.method_results.items():
                for row in rows:
                    values = {f"sweep/{name}/{key}": value for key, value in row.items()
                              if key != "alpha" and value is not None}
                    run.log({"alpha": row["alpha"], **values})
            class_rows = [
                [name, row["samples"], row.get("coverage"), row.get("mean_set_size")]
                for name, row in evaluation.per_class.items()
            ]
            artifacts = {
                "per_class": wandb.Table(
                    columns=["class", "samples", "coverage", "mean_set_size"],
                    data=class_rows,
                ),
            }
            plot = evaluation.cp_dir / "coverage_set_size.png"
            if plot.is_file():
                artifacts["coverage_set_size"] = wandb.Image(str(plot))
            run.log(artifacts)
    except Exception as error:
        warnings.warn(f"W&B evaluation logging failed; local results are saved: {error}", RuntimeWarning)
