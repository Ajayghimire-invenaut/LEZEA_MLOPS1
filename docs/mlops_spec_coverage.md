# LeZeA MLOps Spec Coverage (1.1 – 2.3)

This matrix shows exactly where each requirement is implemented.

| Spec | Requirement | Implementation (Files → Methods) |
|---|---|---|
| **1.1 Experiment Metadata** | Unique experiment ID | `tracker.py` → `ExperimentTracker.__init__` (UUID4 `experiment_id`) |
|  | Experiment name | `tracker.py` → ctor arg `experiment_name` (validated) |
|  | Experiment purpose | `tracker.py` → ctor arg `purpose`; stored in `start()` metadata |
|  | Start/end timestamp | `tracker.py` → `start_time` set at init; `end_time` set in `end()` |
| **1.2 Environment** | Hardware type/specs | `monitoring/env_tags.py` (tags()); `tracker.log_environment_info()` writes artifact |
|  | Firmware | `monitoring/env_tags.py` (extendable fields); logged in `log_environment_info()` |
|  | Software versions (Python, Torch, …) | `env_tags.py` + `log_environment_info()` |
|  | Code version | `tracker.log_git_commit()` |
|  | Data/challenge versions | `tracker.log_dataset_version()` (+ `backends/dvc_backend.py`) |
| **1.3 Constraints** | Runtime duration (global) | `tracker.log_constraints(runtime_limit_seconds=…)` |
|  | Steps & Episodes (per level) | `tracker.log_constraints(step_episode_limits=…)` + `log_training_step(step=…)` |
| **1.4 LeZeA** | Pop sizes (Tasker/Builder) | `tracker.log_lezea_config()` |
|  | Starting networks specs | `tracker.log_lezea_config(start_network_id=…, initialization_scheme=…)` |
| **2.1** | Training algorithm type | `tracker.log_lezea_config(algorithm_type=…)` |
| **2.2** | Starting model network ID | `tracker.log_lezea_config(start_network_id=…)` |
| **2.3** | Hyperparameters | `tracker.log_lezea_config(hyperparameters={…})` |
|  | Seeds & init scheme | `tracker.log_lezea_config(seeds=…, initialization_scheme=…)` |
| **1.5 Training** | Per-step metrics | `tracker.log_training_step()` (auto `delta_loss` if `loss` present) |
|  | Modification trees & paths | `tracker.log_modification_tree()` + artifacts; `log_modification_tree_stats()` |
|  | Modification statistics | `tracker.log_modification_stats()` (alias) |
|  | Process logs (code/MLflow) | MLflow params/artifacts via `start()`, `log_*` methods; local `events.jsonl` |
|  | Data subset splitting | `tracker.log_data_splits()` |
|  | Data usage-rate | `tracker.log_data_usage(sample_ids, epoch)` |
|  | Learning relevance / importance | `tracker.log_data_score(sample_id, before_loss, after_loss)` |
| **1.5.7.1** | Delta loss per cycle | `tracker.log_delta_loss(epoch, loss, prev_loss)` and automatic in `log_training_step()` |
| **1.5.4 Resources** | Compute/Mem/IO per LeZeA part | `monitoring/resource_logger.py` + `tracker.log_component_resources()` / `log_resource()` |
| **1.5.4.1** | Levels: builder/tasker/algorithm/network/layer | `tracker.scope()` helpers + `log_component_resources()` (component_type) |
| **1.5.4.2** | Final resource totals | `tracker._compute_final_resource_summary()` (logged in `end()`) |
| **1.5.5** | Checkpoints (snapshots) | `tracker.log_checkpoint()` (S3 + MLflow artifacts) |
| **1.5.6** | Intermediate/final models | `tracker.log_checkpoint()` + S3 storage |
| **1.6 Results** | Tasker rewards | `tracker.log_results(tasker_rewards=…)` / `tracker.log_reward()` |
|  | Builder rewards | `tracker.log_results(builder_rewards=…)` |
|  | Actions/Outputs | `tracker.log_results(action_outputs=…)` (artifact JSON) |
| **1.7 Business** | Price of resources | `tracker.log_resource_cost()` + optional `CostModel` |
|  | Comments | `tracker.log_comment()` |
|  | Conclusion | `tracker.log_conclusion()` |
|  | Visualizations | `tracker.log_business_metrics(visualization_files=…)` |
| **2 Dataset & Versioning** | Text/Image/Audio modalities | Version tagging via `tracker.log_dataset_version()`; modality captured in `version_metadata` |
|  | DVC version/fingerprint | `backends/dvc_backend.py` (track) + `tracker.log_dataset_version()` |

## Artifact Locations (MLflow)
- `metadata/experiment.json`, `environment/full.json`, `constraints/constraints.json`
- `modifications/step_*.json`, `results/*`, `checkpoints/*`, `datasets/*`
- `resources/summary.json`, `business/*`, `data_usage/*`
