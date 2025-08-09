#!/usr/bin/env python3
"""
LeZeA MLOps - Complete Training Example
======================================

A realistic, end-to-end example that exercises LeZeA MLOps:
- Experiment setup & rich configuration logging
- Environment tagging (hardware, OS, Python, packages)
- DVC dataset versioning
- GPU/system monitoring + Prometheus metrics
- Training/validation simulation with checkpoints, early stopping
- Final evaluation + report artifact + optional model registry

Usage:
  python lezea_mlops/examples/full_training.py \
      [--config config.yaml] [--experiment-name NAME] [--epochs N] \
      [--batch-size N] [--learning-rate LR]
"""

import os
import sys
import json
import time
import yaml
import argparse
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --- LeZeA imports (your modules) -------------------------------------------
from lezea_mlops import ExperimentTracker
from lezea_mlops.monitoring import get_metrics, GPUMonitor, EnvironmentTagger


class LeZeATrainingExample:
    """Complete LeZeA training example with all features"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracker: Optional[ExperimentTracker] = None
        self.metrics = None  # Prometheus collector
        self.gpu_monitor: Optional[GPUMonitor] = None

    # ---------------------- Setup ------------------------------------------------
    def setup_experiment(self):
        print("🔧 Setting up experiment...")

        self.tracker = ExperimentTracker(
            experiment_name=self.config['experiment']['name'],
            description=self.config['experiment']['description'],
            tags=self.config['experiment'].get('tags', {}),
            create_if_not_exists=True,
        )

        # Prometheus metrics
        try:
            self.metrics = get_metrics()
        except Exception as e:
            print(f"⚠️  Prometheus metrics unavailable: {e}")
            self.metrics = None

        # GPU monitoring (runs on demand; can also start background loop if you like)
        try:
            self.gpu_monitor = GPUMonitor(sampling_interval=1.0, history_size=2000)
            print("✅ GPU monitoring initialized")
        except Exception as e:
            print(f"⚠️  GPU monitoring unavailable: {e}")
            self.gpu_monitor = None

        # Environment info + MLflow-compatible tags
        env_tagger = EnvironmentTagger()
        env_info = env_tagger.get_environment_info()
        env_tags = env_tagger.get_mlflow_tags(env_info)

        if hasattr(self.tracker, "log_environment_info"):
            self.tracker.log_environment_info(env_info)
        else:
            # fallback: at least store a few vital bits
            self.tracker.log_params({
                "env.platform": env_info.get("system", {}).get("platform"),
                "env.arch": env_info.get("system", {}).get("architecture"),
            })

        # also set tags if tracker supports it
        if hasattr(self.tracker, "set_tags"):
            self.tracker.set_tags(env_tags)
        else:
            # conservative fallback to params
            self.tracker.log_params({f"env_tag.{k}": v for k, v in env_tags.items()})

        print(f"✅ Experiment '{self.tracker.experiment_name}' initialized")
        print(f"📊 Run ID: {self.tracker.run_id}")
        print("🌐 MLflow UI: http://localhost:5000")

    # ---------------------- Config logging --------------------------------------
    def log_configuration(self):
        print("📝 Logging experiment configuration...")

        # Hyperparameters, model, data basics
        self.tracker.log_params(self.config['model']['hyperparameters'])

        if hasattr(self.tracker, "log_model_architecture"):
            self.tracker.log_model_architecture(self.config['model']['architecture'])
        else:
            self.tracker.log_params({
                f"model.arch.{k}": v for k, v in self.config['model']['architecture'].items()
            })

        if hasattr(self.tracker, "log_dataset_info"):
            self.tracker.log_dataset_info(self.config['data'])
        else:
            self.tracker.log_params({
                f"data.{k}": v for k, v in self.config['data'].items() if not isinstance(v, (list, dict))
            })

        training_cfg = {
            'optimizer': self.config['training']['optimizer'],
            'scheduler': self.config['training'].get('scheduler', {}),
            'regularization': self.config['training'].get('regularization', {}),
            'early_stopping': self.config['training'].get('early_stopping', {}),
        }
        self.tracker.log_params(training_cfg)
        print("✅ Configuration logged")

    # ---------------------- Data loading (simulated) ----------------------------
    def simulate_data_loading(self) -> Dict[str, Any]:
        print("📊 Loading and preparing data...")

        data_cfg = self.config['data']
        load_time = float(np.random.uniform(2.0, 5.0))
        time.sleep(load_time)

        dataset_info = {
            'name': data_cfg['dataset_name'],
            'version': data_cfg['version'],
            'train_samples': int(data_cfg['train_samples']),
            'val_samples': int(data_cfg['val_samples']),
            'test_samples': int(data_cfg['test_samples']),
            'features': int(data_cfg['features']),
            'preprocessing': data_cfg['preprocessing'],
            'load_time_seconds': load_time,
        }

        if hasattr(self.tracker, "log_dataset_metrics"):
            self.tracker.log_dataset_metrics({
                'dataset_size_gb': data_cfg['size_gb'],
                'data_quality_score': 0.95,
                'missing_values_percent': 0.02,
                'data_loading_time': load_time,
            })

        # DVC dataset versioning (aligns with your DVCBackend API)
        try:
            if hasattr(self.tracker, "dvc_backend") and self.tracker.dvc_backend and self.tracker.dvc_backend.available:
                data_path = str(PROJECT_ROOT / "data" / f"{data_cfg['dataset_name']}.parquet")
                Path(data_path).parent.mkdir(parents=True, exist_ok=True)
                # create a dummy file if missing
                if not Path(data_path).exists():
                    Path(data_path).write_text("lezea-demo-data\n")

                version_info = self.tracker.dvc_backend.version_dataset(
                    experiment_id=self.tracker.run_id,
                    dataset_name=data_cfg['dataset_name'],
                    data_paths=[data_path],
                    description=f"Dataset {data_cfg['dataset_name']} v{data_cfg['version']}",
                )
                # keep a record
                if hasattr(self.tracker, "log_params"):
                    self.tracker.log_params({"dvc.version_tag": version_info.get("version_tag", "unknown")})
                print("✅ Dataset versioned with DVC")
        except Exception as e:
            print(f"⚠️  DVC versioning failed: {e}")

        print(f"✅ Data loaded: {dataset_info['train_samples']:,} train, "
              f"{dataset_info['val_samples']:,} val, "
              f"{dataset_info['test_samples']:,} test")
        return dataset_info

    # ---------------------- Model init (simulated) ------------------------------
    def simulate_model_initialization(self) -> Dict[str, Any]:
        print("🧠 Initializing model...")
        arch = self.config['model']['architecture']
        init_time = float(np.random.uniform(1.0, 3.0))
        time.sleep(init_time)

        total_params = (
            arch['hidden_size'] * arch['vocab_size'] +                                  # embeddings
            arch['num_layers'] * arch['hidden_size'] * arch['hidden_size'] * 4 +        # transformer blocks
            arch['hidden_size'] * arch['vocab_size']                                     # output
        )

        model_info = {
            'total_parameters': int(total_params),
            'trainable_parameters': int(total_params * 0.95),
            'model_size_mb': (total_params * 4) / (1024 * 1024),  # float32
            'initialization_time': init_time,
            'architecture_type': arch['type'],
        }

        if hasattr(self.tracker, "log_model_info"):
            self.tracker.log_model_info(model_info)
        else:
            self.tracker.log_params({f"model.{k}": v for k, v in model_info.items()})

        # seed Prometheus model metrics row (optional)
        if self.metrics:
            self.metrics.record_model_metrics(
                experiment_id=self.tracker.run_id,
                model_type=arch['type'],
            )

        print(f"✅ Model initialized: {model_info['total_parameters']:,} params "
              f"({model_info['model_size_mb']:.1f} MB)")
        return model_info

    # ---------------------- Training loop (simulated) ---------------------------
    def simulate_training_loop(self, dataset_info: Dict[str, Any], model_info: Dict[str, Any]) -> float:
        print("🚀 Starting training loop...")

        tr_cfg = self.config['training']
        hp = self.config['model']['hyperparameters']
        epochs = int(tr_cfg['epochs'])
        batch_size = int(hp['batch_size'])
        learning_rate = float(hp['learning_rate'])

        steps_per_epoch = max(1, dataset_info['train_samples'] // batch_size)
        total_steps = epochs * steps_per_epoch
        print(f"📈 Training for {epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps")

        best_val_acc = 0.0
        no_improve = 0
        global_step = 0

        for epoch in range(epochs):
            e_start = time.time()
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")

            train_metrics = self._simulate_epoch(
                phase="train",
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                global_step=global_step,
                learning_rate=learning_rate,
            )
            global_step += steps_per_epoch

            val_metrics = self._simulate_epoch(
                phase="val",
                epoch=epoch,
                steps_per_epoch=max(1, dataset_info['val_samples'] // batch_size),
                global_step=global_step,
            )

            epoch_time = time.time() - e_start
            combined = {
                **train_metrics, **val_metrics,
                "epoch": epoch,
                "learning_rate": learning_rate * (0.95 ** epoch),
                "epoch_time_sec": epoch_time,
            }

            self.tracker.log_metrics(combined, step=global_step)

            # GPU + system usage snapshot -> tracker + Prometheus
            self._log_gpu_and_system_metrics()

            if self.metrics:
                # approximate step time + throughput
                self.metrics.record_training_step(
                    experiment_id=self.tracker.run_id,
                    model_type=self.config['model']['architecture']['type'],
                    step_time=epoch_time / max(1, steps_per_epoch),
                    loss=train_metrics['train_loss'],
                    samples_per_sec=(batch_size * steps_per_epoch) / max(1e-6, epoch_time),
                )
                self.metrics.record_epoch_completion(
                    experiment_id=self.tracker.run_id,
                    model_type=self.config['model']['architecture']['type'],
                    epoch_time=epoch_time,
                )

            # checkpointing
            if (epoch % tr_cfg.get('checkpoint_every', 5)) == 0:
                self._save_checkpoint(epoch, combined)

            # early stopping check
            val_acc = val_metrics['val_accuracy']
            if val_acc > best_val_acc + tr_cfg.get('early_stopping', {}).get('min_delta', 0.0):
                best_val_acc = val_acc
                no_improve = 0
                self._save_best_model(epoch, combined)
            else:
                no_improve += 1

            # epoch summary
            print(f"✅ Epoch {epoch + 1} in {epoch_time:.1f}s "
                  f"| Train: loss={train_metrics['train_loss']:.4f}, acc={train_metrics['train_accuracy']:.4f} "
                  f"| Val: loss={val_metrics['val_loss']:.4f}, acc={val_acc:.4f}")

            # early stop
            if no_improve >= tr_cfg.get('early_stopping', {}).get('patience', 10):
                print(f"🛑 Early stopping after {no_improve} epochs w/o improvement")
                break

            # optional manual LR decay
            if epoch > 0 and (epoch % tr_cfg.get('lr_decay_every', 9999) == 0):
                learning_rate *= tr_cfg.get('lr_decay_factor', 1.0)
                print(f"📉 LR decayed to {learning_rate:.6f}")

        print(f"🎯 Training finished | Best val acc: {best_val_acc:.4f}")
        return best_val_acc

    def _simulate_epoch(self, phase: str, epoch: int, steps_per_epoch: int,
                        global_step: int, learning_rate: Optional[float] = None) -> Dict[str, float]:
        """Simulate metrics evolution with noise."""
        is_train = (phase == "train")
        if is_train:
            base_loss = 2.0 * np.exp(-epoch * 0.15) + 0.1
            base_acc = min(0.98, 1.0 - np.exp(-epoch * 0.2))
        else:
            base_loss = 2.2 * np.exp(-epoch * 0.12) + 0.15
            base_acc = min(0.95, 0.95 - np.exp(-epoch * 0.18))

        loss = float(base_loss + np.random.normal(0, 0.05))
        acc = float(np.clip(base_acc + np.random.normal(0, 0.02), 0.0, 1.0))

        # lightweight per-step logging (10 checkpoints per epoch)
        if is_train and steps_per_epoch > 50 and hasattr(self.tracker, "log_metrics"):
            stride = max(1, steps_per_epoch // 10)
            for step in range(0, steps_per_epoch, stride):
                step_metrics = {
                    f'{phase}_loss_step': float(loss + np.random.normal(0, 0.02)),
                    f'{phase}_accuracy_step': float(np.clip(acc + np.random.normal(0, 0.01), 0.0, 1.0)),
                    'step': global_step + step,
                }
                if learning_rate is not None:
                    step_metrics['learning_rate'] = float(learning_rate)
                self.tracker.log_metrics(step_metrics, step=global_step + step)
                time.sleep(0.01)

        # simulate compute time for the epoch
        time.sleep(min(2.0, steps_per_epoch * 0.001))

        # return epoch aggregates
        if is_train:
            return {
                'train_loss': loss,
                'train_accuracy': acc,
                'train_perplexity': float(np.exp(loss)),
                'train_f1_score': float(np.clip(acc * 0.95 + np.random.normal(0, 0.01), 0.0, 1.0)),
            }
        else:
            return {
                'val_loss': loss,
                'val_accuracy': acc,
                'val_perplexity': float(np.exp(loss)),
                'val_f1_score': float(np.clip(acc * 0.95 + np.random.normal(0, 0.01), 0.0, 1.0)),
            }

    # ---------------------- GPU + system logging helper -------------------------
    def _log_gpu_and_system_metrics(self):
        if not self.gpu_monitor:
            return

        snapshot = self.gpu_monitor.get_current_usage()
        gpu_list = snapshot.get('gpu_devices', []) or []

        # log to tracker (first GPU summary)
        if gpu_list and hasattr(self.tracker, "log_resource_usage"):
            g0 = gpu_list[0]
            self.tracker.log_resource_usage({
                'gpu_utilization_percent': g0.get('utilization_percent', g0.get('utilization', 0)),
                'gpu_memory_used_mb': g0.get('memory_used_mb', 0),
                'gpu_temperature_c': g0.get('temperature_c', g0.get('temperature', 0)),
            })

        # normalize for Prometheus update_gpu_metrics API
        if self.metrics and gpu_list:
            prom_ready = []
            for g in gpu_list:
                mem_used_mb = g.get('memory_used_mb') or g.get('memory_allocated_mb')
                mem_total_mb = g.get('memory_total_mb')
                prom_ready.append({
                    'id': g.get('device_id', g.get('id', 0)),
                    'name': g.get('name', 'unknown'),
                    'utilization': g.get('utilization_percent', g.get('utilization', 0)),
                    'memory_used': (mem_used_mb or 0) * 1024 * 1024,
                    'memory_total': (mem_total_mb or 0) * 1024 * 1024,
                    'temperature': g.get('temperature_c', g.get('temperature', 0)),
                    'power_draw': g.get('power_usage_w', g.get('power_draw_w', 0)),
                })
            self.metrics.update_gpu_metrics(prom_ready)

    # ---------------------- Artifacts ------------------------------------------
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        info = {
            'epoch': epoch,
            'model_state': f"checkpoint_epoch_{epoch}.pth",
            'optimizer_state': f"optimizer_epoch_{epoch}.pth",
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(info, f, indent=2)
            tmp = f.name
        try:
            self.tracker.log_artifact(tmp, f"checkpoints/checkpoint_epoch_{epoch}.json")
            print(f"💾 Checkpoint saved (epoch {epoch})")
        finally:
            try: os.unlink(tmp)
            except OSError: pass

    def _save_best_model(self, epoch: int, metrics: Dict[str, float]):
        payload = {
            'epoch': epoch,
            'metrics': metrics,
            'model_architecture': self.config['model']['architecture'],
            'hyperparameters': self.config['model']['hyperparameters'],
            'timestamp': datetime.now().isoformat(),
            'best_model': True,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(payload, f, indent=2)
            tmp = f.name
        try:
            self.tracker.log_artifact(tmp, "models/best_model.json")
            # optional: register to model registry if supported
            if hasattr(self.tracker, "register_model"):
                model_name = f"{self.config['experiment']['name']}_model"
                self.tracker.register_model(
                    model_name=model_name,
                    model_path="models/best_model.json",
                    description=f"Best model @ epoch {epoch} (val_acc={metrics.get('val_accuracy', 0):.4f})",
                )
                print(f"🏆 Best model saved & registered: {model_name}")
            else:
                print("🏆 Best model saved")
        finally:
            try: os.unlink(tmp)
            except OSError: pass

    # ---------------------- Evaluation & report ---------------------------------
    def run_evaluation(self, best_accuracy: float) -> Dict[str, float]:
        print("\n🧪 Running model evaluation...")
        test = {
            'test_accuracy': float(np.clip(best_accuracy * 0.98 + np.random.normal(0, 0.01), 0.0, 1.0)),
            'test_loss': float(0.15 + np.random.normal(0, 0.02)),
            'test_f1_score': float(np.clip(best_accuracy * 0.96 + np.random.normal(0, 0.01), 0.0, 1.0)),
            'test_precision': float(np.clip(best_accuracy * 0.97 + np.random.normal(0, 0.01), 0.0, 1.0)),
            'test_recall': float(np.clip(best_accuracy * 0.95 + np.random.normal(0, 0.01), 0.0, 1.0)),
            'bleu_score': float(np.clip(85 + np.random.normal(0, 5), 0, 100)),
            'rouge_l': float(np.clip(0.8 + np.random.normal(0, 0.05), 0, 1.0)),
        }
        test['perplexity'] = float(np.exp(test['test_loss']))

        self.tracker.log_metrics(test)

        if self.metrics:
            self.metrics.record_model_metrics(
                experiment_id=self.tracker.run_id,
                model_type=self.config['model']['architecture']['type'],
                accuracy=test['test_accuracy'],
                precision=test['test_precision'],
                recall=test['test_recall'],
                f1_score=test['test_f1_score'],
                dataset='test',
            )

        print("✅ Evaluation completed:")
        for k, v in test.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
        return test

    def generate_final_report(self, test_metrics: Dict[str, float]) -> Dict[str, Any]:
        print("\n📋 Generating final report...")
        run_data = {}
        if hasattr(self.tracker, "get_run_data"):
            run_data = self.tracker.get_run_data()

        start_time = run_data.get('start_time', datetime.now().isoformat())
        duration_min = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds() / 60.0

        best_val = 0.0
        metrics_blob = run_data.get('metrics', {}) if isinstance(run_data.get('metrics', {}), dict) else {}
        for k, v in metrics_blob.items():
            if 'val_accuracy' in k:
                try:
                    best_val = max(best_val, float(v))
                except Exception:
                    pass

        report = {
            'experiment_summary': {
                'name': getattr(self.tracker, "experiment_name", "unknown"),
                'run_id': getattr(self.tracker, "run_id", "unknown"),
                'start_time': start_time,
                'end_time': datetime.now().isoformat(),
                'duration_minutes': duration_min,
            },
            'configuration': self.config,
            'final_metrics': test_metrics,
            'best_validation_accuracy': best_val,
            'total_parameters': run_data.get('model_info', {}).get('total_parameters', 0),
            'artifacts': run_data.get('artifacts', []),
            'tags': run_data.get('tags', {}),
            'environment': run_data.get('environment', {}),
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(report, f, indent=2)
            tmp = f.name
        try:
            self.tracker.log_artifact(tmp, "reports/final_report.json")
            print("✅ Final report saved")
        finally:
            try: os.unlink(tmp)
            except OSError: pass

        print("\n" + "=" * 60)
        print("🎯 EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Experiment: {report['experiment_summary']['name']}")
        print(f"Run ID: {report['experiment_summary']['run_id']}")
        print(f"Duration: {report['experiment_summary']['duration_minutes']:.1f} minutes")
        print(f"Best Val Accuracy: {report['best_validation_accuracy']:.4f}")
        print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        print(f"Model Parameters: {report['total_parameters']:,}")
        print(f"Artifacts: {len(report['artifacts'])}")
        print("=" * 60)
        return report

    # ---------------------- Cleanup --------------------------------------------
    def cleanup(self):
        print("\n🧹 Cleaning up...")
        try:
            if self.tracker:
                self.tracker.finish_run(status="FINISHED")
            if self.metrics:
                # optional: mark experiment completion
                self.metrics.record_experiment_completion(
                    model_type=self.config['model']['architecture']['type'],
                    success=True,
                )
            print("✅ Experiment finished successfully")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
            if self.tracker:
                self.tracker.finish_run(status="FAILED")


# ---------------------- Config loading ----------------------------------------
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    default = {
        'experiment': {
            'name': 'lezea_full_training_demo',
            'description': 'Complete LeZeA MLOps training demonstration with all features',
            'tags': {'team': 'ml-research', 'priority': 'high', 'framework': 'pytorch', 'model_family': 'transformer'},
        },
        'model': {
            'architecture': {
                'type': 'transformer', 'num_layers': 12, 'hidden_size': 768,
                'num_heads': 12, 'vocab_size': 50000, 'max_sequence_length': 512, 'dropout_rate': 0.1,
            },
            'hyperparameters': {
                'learning_rate': 1e-4, 'batch_size': 32, 'weight_decay': 0.01,
                'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'max_grad_norm': 1.0, 'warmup_steps': 1000,
            },
        },
        'data': {
            'dataset_name': 'lezea_training_data', 'version': '1.2.0',
            'train_samples': 100_000, 'val_samples': 10_000, 'test_samples': 5_000,
            'features': 768, 'size_gb': 2.5, 'preprocessing': ['tokenization', 'normalization', 'augmentation'],
        },
        'training': {
            'epochs': 20, 'optimizer': 'adamw',
            'scheduler': {'type': 'cosine', 'warmup_epochs': 2, 'min_lr': 1e-6},
            'regularization': {'dropout': 0.1, 'weight_decay': 0.01, 'label_smoothing': 0.1},
            'early_stopping': {'patience': 5, 'min_delta': 0.001},
            'checkpoint_every': 3, 'lr_decay_every': 8, 'lr_decay_factor': 0.8,
        },
    }

    if config_path and os.path.exists(config_path):
        print(f"📂 Loading config from {config_path}")
        with open(config_path, 'r') as f:
            custom = yaml.safe_load(f) if config_path.endswith(('.yaml', '.yml')) else json.load(f)

        def merge(a: Dict[str, Any], b: Dict[str, Any]):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k] = v

        merge(default, custom)
    return default


# ---------------------- Entrypoint --------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='LeZeA MLOps Complete Training Example')
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--experiment-name', type=str, help='Override experiment name')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.experiment_name: config['experiment']['name'] = args.experiment_name
    if args.epochs:          config['training']['epochs'] = args.epochs
    if args.batch_size:      config['model']['hyperparameters']['batch_size'] = args.batch_size
    if args.learning_rate:   config['model']['hyperparameters']['learning_rate'] = args.learning_rate

    print("🚀 LeZeA MLOps - Complete Training Example")
    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Model: {config['model']['architecture']['type']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['model']['hyperparameters']['batch_size']}")
    print("=" * 60)

    trainer = LeZeATrainingExample(config)
    try:
        trainer.setup_experiment()
        trainer.log_configuration()
        dataset_info = trainer.simulate_data_loading()
        model_info = trainer.simulate_model_initialization()
        best_acc = trainer.simulate_training_loop(dataset_info, model_info)
        test_metrics = trainer.run_evaluation(best_acc)
        trainer.generate_final_report(test_metrics)
        print("\n🎉 Training completed successfully!")
        print("🌐 MLflow:     http://localhost:5000")
        print("📊 Prometheus: http://localhost:9090")
        print("📈 Grafana:    http://localhost:3000")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if trainer.tracker:
            trainer.tracker.finish_run(status="FAILED")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
