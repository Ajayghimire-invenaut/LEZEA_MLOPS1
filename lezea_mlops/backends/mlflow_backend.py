# lezea_mlops/backends/mlflow_backend.py
from __future__ import annotations

import os
import json
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover — environment guard
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False


def _stringify(value: Any, max_len: int = 500) -> str:
    if isinstance(value, (dict, list)):
        s = json.dumps(value)
    elif value is None:
        s = "null"
    else:
        s = str(value)
    return s if len(s) <= max_len else (s[: max_len - 3] + "...")


class MLflowBackend:
    """Thin, stable wrapper around MLflow for LeZeA."""

    def __init__(self, config) -> None:
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow is not available. Install with: pip install mlflow")

        self.config = config
        self.mlflow_config = config.get_mlflow_config()

        # Configure tracking URI
        mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])

        # State
        self.current_experiment: Optional[str] = None
        self.current_experiment_id: Optional[str] = None
        self.current_run = None
        self.current_run_id: Optional[str] = None
        self._pending_experiment_tags: Dict[str, str] = {}
        self.available: bool = False

        # Verify connection early
        self._verify_connection()
        self.available = True
        print(f"✅ MLflow backend connected: {self.mlflow_config['tracking_uri']}")

    # ------------------------------------------------------------------
    # Connection / experiment management
    # ------------------------------------------------------------------
    def _verify_connection(self) -> None:
        try:
            # If server is reachable, this should succeed
            _ = mlflow.search_experiments(max_results=1)
        except Exception as e:  # pragma: no cover
            raise ConnectionError(f"Failed to connect to MLflow: {e}")

    def ping(self) -> bool:
        try:
            _ = mlflow.search_experiments(max_results=1)
            return True
        except Exception:
            return False

    def create_experiment(
        self,
        experiment_name: str,
        lezea_experiment_id: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create or reuse an experiment and mark it as current."""
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                mlflow_experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location or self.mlflow_config.get("default_artifact_root"),
                    tags={k: _stringify(v) for k, v in (tags or {}).items()} or None,
                )
                print(f"📁 Created MLflow experiment: {experiment_name} (ID: {mlflow_experiment_id})")
            else:
                mlflow_experiment_id = exp.experiment_id
                print(f"📂 Using existing MLflow experiment: {experiment_name} (ID: {mlflow_experiment_id})")

            mlflow.set_experiment(experiment_name)
            self.current_experiment = experiment_name
            self.current_experiment_id = mlflow_experiment_id

            # Stash tags for subsequent runs and try to write as experiment tags
            self._add_experiment_tags(mlflow_experiment_id, lezea_experiment_id, tags)
            return mlflow_experiment_id
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to create/get experiment: {e}")
            raise

    def _add_experiment_tags(
        self, mlflow_experiment_id: str, lezea_experiment_id: str, additional_tags: Optional[Dict[str, str]] = None
    ) -> None:
        try:
            base = {
                "lezea_experiment_id": lezea_experiment_id,
                "lezea_version": "1.0.0",
                "created_by": "lezea_mlops",
                "creation_timestamp": datetime.now().isoformat(),
            }
            if additional_tags:
                base.update({str(k): _stringify(v) for k, v in additional_tags.items()})
            self._pending_experiment_tags.update(base)

            # Best-effort: set experiment tags when server allows it
            try:
                client = MlflowClient()
                for k, v in base.items():
                    client.set_experiment_tag(mlflow_experiment_id, k, str(v))
            except Exception:
                pass
        except Exception as e:  # pragma: no cover
            print(f"⚠️ Failed to add experiment tags: {e}")

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None, nested: bool = False) -> str:
        try:
            if self.current_run is not None:
                # Idempotency: don't silently stack runs
                print("ℹ️ Ending previous active run before starting a new one")
                self.end_run(status="FINISHED")

            run_tags = {
                "mlflow.runName": run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "lezea.created_by": "lezea_mlops",
                "lezea.start_time": datetime.now().isoformat(),
            }
            if self._pending_experiment_tags:
                run_tags.update(self._pending_experiment_tags)
            if tags:
                run_tags.update({str(k): _stringify(v) for k, v in tags.items()})

            run = mlflow.start_run(run_name=run_name, tags=run_tags, nested=nested)
            self.current_run = run
            self.current_run_id = run.info.run_id
            print(f"🏃 Started MLflow run: {run_name or 'unnamed'} (ID: {self.current_run_id[:8]}...)")
            return self.current_run_id
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to start run: {e}")
            raise

    def end_run(self, status: str = "FINISHED") -> None:
        try:
            if self.current_run is None:
                print("⚠️ No active run to end")
                return
            # put an end timestamp metric for convenience
            try:
                self.log_metric("run_end_timestamp", datetime.now().timestamp())
            except Exception:
                pass
            mlflow.end_run(status=status)
            print(f"🏁 Ended MLflow run with status: {status}")
        finally:
            self.current_run = None
            self.current_run_id = None

    @contextmanager
    def child_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Context manager for a nested child run."""
        self.start_run(run_name=run_name, tags=tags, nested=True)
        try:
            yield self.current_run_id
        finally:
            self.end_run(status="FINISHED")

    # ------------------------------------------------------------------
    # Logging helpers (guarded + portable)
    # ------------------------------------------------------------------
    def log_param(self, key: str, value: Any) -> None:
        try:
            mlflow.log_param(key, _stringify(value))
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to log param {key}: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        try:
            processed = {k: _stringify(v) for k, v in params.items()}
            if processed:
                mlflow.log_params(processed)
                print(f"📝 Logged {len(processed)} parameters")
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to log params: {e}")

    def log_metric(self, key: str, value: Union[int, float], step: Optional[int] = None) -> None:
        try:
            if not isinstance(value, (int, float)):
                value = float(value)
            mlflow.log_metric(key, value, step=step)
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to log metric {key}: {e}")

    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        try:
            numeric = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    numeric[k] = v
                else:
                    try:
                        numeric[k] = float(v)
                    except Exception:
                        print(f"⚠️ Skipping non-numeric metric {k}: {v}")
            if numeric:
                mlflow.log_metrics(numeric, step=step)
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to log metrics: {e}")

    def set_tag(self, key: str, value: str) -> None:
        try:
            mlflow.set_tag(key, _stringify(value))
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to set tag {key}: {e}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        try:
            mlflow.set_tags({k: _stringify(v) for k, v in tags.items()})
            print(f"🏷️ Set {len(tags)} tags")
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to set tags: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        try:
            if not os.path.exists(local_path):
                print(f"⚠️ Artifact file not found: {local_path}")
                return
            mlflow.log_artifact(local_path, artifact_path)
            print(f"📎 Logged artifact: {os.path.basename(local_path)}")
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to log artifact: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        try:
            if not os.path.isdir(local_dir):
                print(f"⚠️ Artifact directory not found: {local_dir}")
                return
            mlflow.log_artifacts(local_dir, artifact_path)
            print(f"📁 Logged artifacts from: {local_dir}")
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to log artifacts: {e}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact under the exact filename."""
        try:
            if hasattr(mlflow, "log_dict"):
                mlflow.log_dict(dictionary, artifact_file=artifact_file)
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    target = Path(tmp) / artifact_file
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(dictionary, indent=2, default=str), encoding="utf-8")
                    mlflow.log_artifacts(tmp)
            print(f"📄 Logged dictionary as: {artifact_file}")
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to log dictionary: {e}")

    def log_text(self, text: str, artifact_file: str) -> None:
        """Log a text artifact under the exact filename."""
        try:
            if hasattr(mlflow, "log_text"):
                mlflow.log_text(text, artifact_file=artifact_file)
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    target = Path(tmp) / artifact_file
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text, encoding="utf-8")
                    mlflow.log_artifacts(tmp)
            print(f"📝 Logged text as: {artifact_file}")
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to log text: {e}")

    def log_figure(self, figure, artifact_file: str) -> None:
        """Log a matplotlib figure under the exact filename."""
        try:
            if hasattr(mlflow, "log_figure"):
                mlflow.log_figure(figure, artifact_file)
                print(f"🖼️ Logged figure as: {artifact_file}")
                return
            # Fallback: save PNG to temp and log_artifacts
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / artifact_file
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.suffix == "":
                    target = target.with_suffix(".png")
                figure.savefig(str(target), bbox_inches="tight")
                mlflow.log_artifacts(tmp)
            print(f"🖼️ Logged figure as: {artifact_file}")
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to log figure: {e}")

    def log_artifact_with_meta(self, local_path: str, meta: Dict[str, Any], artifact_path: Optional[str] = None) -> None:
        """Upload a file and a sibling .meta.json with provided metadata."""
        try:
            self.log_artifact(local_path, artifact_path)
            name = os.path.basename(local_path)
            meta_file = f"{artifact_path + '/' if artifact_path else ''}{name}.meta.json"
            self.log_dict(meta, meta_file)
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to log artifact metadata: {e}")

    # ------------------------------------------------------------------
    # Registry & queries
    # ------------------------------------------------------------------
    def register_model(
        self, model_name: str, model_path: Optional[str] = None, description: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        try:
            model_uri = f"runs:/{self.current_run_id}/model" if model_path is None else model_path
            result = mlflow.register_model(model_uri, model_name, tags={k: _stringify(v) for k, v in (tags or {}).items()} or None)
            version = result.version
            if description:
                try:
                    client = MlflowClient()
                    client.update_model_version(name=model_name, version=version, description=description)
                except Exception:
                    pass
            print(f"🏆 Registered model: {model_name} v{version}")
            return version
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to register model: {e}")
            return None

    def search_runs(
        self, experiment_ids: Optional[List[str]] = None, filter_string: str = "", order_by: Optional[List[str]] = None, max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        try:
            if experiment_ids is None and self.current_experiment_id:
                experiment_ids = [self.current_experiment_id]
            df = mlflow.search_runs(experiment_ids=experiment_ids, filter_string=filter_string, order_by=order_by, max_results=max_results)
            # Return a list of dicts for portability
            return [row._asdict() if hasattr(row, "_asdict") else row.to_dict() for _, row in df.iterrows()]
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to search runs: {e}")
            return []

    def get_run_info(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            target = run_id or self.current_run_id
            if not target:
                return {}
            run = mlflow.get_run(target)
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": dict(run.data.tags),
                "artifact_uri": run.info.artifact_uri,
            }
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to get run info: {e}")
            return {}

    # ------------------------------------------------------------------
    # URLs & cleanup
    # ------------------------------------------------------------------
    def get_experiment_url(self) -> str:
        base = self.mlflow_config["tracking_uri"].rstrip("/")
        if self.current_experiment_id:
            return f"{base}/#/experiments/{self.current_experiment_id}"
        return base

    def get_run_url(self) -> str:
        base = self.mlflow_config["tracking_uri"].rstrip("/")
        if self.current_run_id and self.current_experiment_id:
            return f"{base}/#/experiments/{self.current_experiment_id}/runs/{self.current_run_id}"
        return self.get_experiment_url()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_run is not None:
            self.end_run("KILLED")
        print("🧹 Cleaned up MLflow backend")