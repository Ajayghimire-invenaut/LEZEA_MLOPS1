#!/usr/bin/env python3
"""
LeZeA MLOps - Complete Training Example
======================================

Comprehensive example showing all LeZeA MLOps features:
- Advanced experiment configuration
- Comprehensive logging and monitoring
- Resource tracking and optimization
- Model versioning and deployment
- Integration with all backends

This is the complete example Marcus can use as a template for production training.

Usage:
    python examples/full_training.py [--config config.yaml] [--experiment-name NAME]
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
from typing import Dict, List, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lezea_mlops import ExperimentTracker
from lezea_mlops.monitoring import get_metrics, GPUMonitor, EnvironmentTagger

class LeZeATrainingExample:
    """Complete LeZeA training example with all features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracker = None
        self.metrics_collector = None
        self.gpu_monitor = None
        
    def setup_experiment(self):
        """Initialize experiment tracking and monitoring"""
        print("🔧 Setting up experiment...")
        
        # Initialize experiment tracker with advanced configuration
        self.tracker = ExperimentTracker(
            experiment_name=self.config['experiment']['name'],
            description=self.config['experiment']['description'],
            tags=self.config['experiment'].get('tags', {}),
            create_if_not_exists=True
        )
        
        # Initialize metrics collection
        self.metrics_collector = get_metrics()
        
        # Initialize GPU monitoring if available
        try:
            self.gpu_monitor = GPUMonitor()
            print("✅ GPU monitoring initialized")
        except Exception as e:
            print(f"⚠️  GPU monitoring unavailable: {e}")
            self.gpu_monitor = None
        
        # Log environment information
        env_tagger = EnvironmentTagger()
        env_info = env_tagger.get_complete_environment()
        self.tracker.log_environment_info(env_info)
        
        print(f"✅ Experiment '{self.tracker.experiment_name}' initialized")
        print(f"📊 Run ID: {self.tracker.run_id}")
        print(f"🌐 MLflow UI: http://localhost:5000")
        
    def log_configuration(self):
        """Log complete experiment configuration"""
        print("📝 Logging experiment configuration...")
        
        # Log hyperparameters
        self.tracker.log_params(self.config['model']['hyperparameters'])
        
        # Log model architecture
        self.tracker.log_model_architecture(self.config['model']['architecture'])
        
        # Log dataset information
        self.tracker.log_dataset_info(self.config['data'])
        
        # Log training configuration
        training_config = {
            'optimizer': self.config['training']['optimizer'],
            'scheduler': self.config['training'].get('scheduler', {}),
            'regularization': self.config['training'].get('regularization', {}),
            'early_stopping': self.config['training'].get('early_stopping', {})
        }
        self.tracker.log_params(training_config)
        
        print("✅ Configuration logged")
    
    def simulate_data_loading(self) -> Dict[str, Any]:
        """Simulate data loading with monitoring"""
        print("📊 Loading and preparing data...")
        
        data_config = self.config['data']
        
        # Simulate data loading time
        load_time = np.random.uniform(2.0, 5.0)
        time.sleep(load_time)
        
        # Create simulated dataset metadata
        dataset_info = {
            'name': data_config['dataset_name'],
            'version': data_config['version'],
            'train_samples': data_config['train_samples'],
            'val_samples': data_config['val_samples'],
            'test_samples': data_config['test_samples'],
            'features': data_config['features'],
            'preprocessing': data_config['preprocessing'],
            'load_time_seconds': load_time
        }
        
        # Log dataset metrics
        self.tracker.log_dataset_metrics({
            'dataset_size_gb': data_config['size_gb'],
            'data_quality_score': 0.95,
            'missing_values_percent': 0.02,
            'data_loading_time': load_time
        })
        
        # Register dataset version with DVC
        try:
            self.tracker.dvc_backend.track_dataset(
                dataset_name=data_config['dataset_name'],
                version=data_config['version'],
                file_path=f"data/{data_config['dataset_name']}.parquet",
                metadata=dataset_info
            )
            print("✅ Dataset version tracked with DVC")
        except Exception as e:
            print(f"⚠️  DVC tracking failed: {e}")
        
        print(f"✅ Data loaded: {dataset_info['train_samples']:,} train, "
              f"{dataset_info['val_samples']:,} validation, "
              f"{dataset_info['test_samples']:,} test samples")
        
        return dataset_info
    
    def simulate_model_initialization(self) -> Dict[str, Any]:
        """Simulate model initialization with architecture logging"""
        print("🧠 Initializing model...")
        
        model_config = self.config['model']
        
        # Simulate model creation time
        init_time = np.random.uniform(1.0, 3.0)
        time.sleep(init_time)
        
        # Calculate model parameters
        architecture = model_config['architecture']
        total_params = (
            architecture['hidden_size'] * architecture['vocab_size'] +  # Embedding
            architecture['num_layers'] * architecture['hidden_size'] * architecture['hidden_size'] * 4 +  # Transformer layers
            architecture['hidden_size'] * architecture['vocab_size']  # Output layer
        )
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': int(total_params * 0.95),  # Some frozen params
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'initialization_time': init_time,
            'architecture_type': architecture['type']
        }
        
        # Log model information
        self.tracker.log_model_info(model_info)
        
        # Log to Prometheus if available
        if self.metrics_collector:
            self.metrics_collector.record_model_metrics(
                experiment_id=self.tracker.run_id,
                model_type=architecture['type']
            )
        
        print(f"✅ Model initialized: {total_params:,} parameters ({model_info['model_size_mb']:.1f} MB)")
        
        return model_info
    
    def simulate_training_loop(self, dataset_info: Dict[str, Any], model_info: Dict[str, Any]):
        """Simulate complete training loop with comprehensive monitoring"""
        print("🚀 Starting training loop...")
        
        training_config = self.config['training']
        num_epochs = training_config['epochs']
        batch_size = self.config['model']['hyperparameters']['batch_size']
        learning_rate = self.config['model']['hyperparameters']['learning_rate']
        
        # Calculate steps per epoch
        steps_per_epoch = dataset_info['train_samples'] // batch_size
        total_steps = num_epochs * steps_per_epoch
        
        print(f"📈 Training for {num_epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps")
        
        # Training state
        best_val_accuracy = 0.0
        no_improvement_count = 0
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            epoch_metrics = self._simulate_epoch(
                phase="train",
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                global_step=global_step,
                learning_rate=learning_rate
            )
            
            global_step += steps_per_epoch
            
            # Validation phase
            val_metrics = self._simulate_epoch(
                phase="validation",
                epoch=epoch,
                steps_per_epoch=dataset_info['val_samples'] // batch_size,
                global_step=global_step
            )
            
            # Combine metrics
            combined_metrics = {**epoch_metrics, **val_metrics}
            combined_metrics['epoch'] = epoch
            combined_metrics['learning_rate'] = learning_rate * (0.95 ** epoch)  # LR decay
            
            # Log epoch metrics
            self.tracker.log_metrics(combined_metrics, step=global_step)
            
            # Resource monitoring
            if self.gpu_monitor:
                gpu_stats = self.gpu_monitor.get_gpu_stats()
                if gpu_stats:
                    self.tracker.log_resource_usage({
                        'gpu_utilization': gpu_stats[0].get('utilization_percent', 0),
                        'gpu_memory_used_mb': gpu_stats[0].get('memory_used_mb', 0),
                        'gpu_temperature': gpu_stats[0].get('temperature', 0)
                    })
                    
                    # Log to Prometheus
                    if self.metrics_collector:
                        self.metrics_collector.update_gpu_metrics(gpu_stats)
            
            # Log to Prometheus
            if self.metrics_collector:
                epoch_time = time.time() - epoch_start_time
                self.metrics_collector.record_training_step(
                    experiment_id=self.tracker.run_id,
                    model_type=self.config['model']['architecture']['type'],
                    step_time=epoch_time / steps_per_epoch,
                    loss=epoch_metrics['train_loss'],
                    samples_per_sec=batch_size * steps_per_epoch / epoch_time
                )
                
                self.metrics_collector.record_epoch_completion(
                    experiment_id=self.tracker.run_id,
                    model_type=self.config['model']['architecture']['type'],
                    epoch_time=epoch_time
                )
            
            # Model checkpointing
            if epoch % training_config.get('checkpoint_every', 5) == 0:
                self._save_checkpoint(epoch, combined_metrics)
            
            # Early stopping check
            val_accuracy = val_metrics['val_accuracy']
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                no_improvement_count = 0
                self._save_best_model(epoch, combined_metrics)
            else:
                no_improvement_count += 1
            
            # Print epoch summary
            print(f"✅ Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.1f}s")
            print(f"   Train Loss: {epoch_metrics['train_loss']:.4f}, "
                  f"Train Acc: {epoch_metrics['train_accuracy']:.4f}")
            print(f"   Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.4f}")
            
            # Early stopping
            early_stop_patience = training_config.get('early_stopping', {}).get('patience', 10)
            if no_improvement_count >= early_stop_patience:
                print(f"🛑 Early stopping triggered after {no_improvement_count} epochs without improvement")
                break
            
            # Learning rate decay
            if epoch > 0 and epoch % training_config.get('lr_decay_every', 10) == 0:
                learning_rate *= training_config.get('lr_decay_factor', 0.8)
                print(f"📉 Learning rate decayed to {learning_rate:.6f}")
        
        print(f"🎯 Training completed! Best validation accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy
    
    def _simulate_epoch(self, phase: str, epoch: int, steps_per_epoch: int, 
                       global_step: int, learning_rate: float = None) -> Dict[str, float]:
        """Simulate a single epoch of training or validation"""
        
        is_training = phase == "train"
        
        # Simulate epoch with realistic loss curves
        if is_training:
            # Training loss decreases with some noise
            base_loss = 2.0 * np.exp(-epoch * 0.15) + 0.1
            base_accuracy = min(0.98, 1.0 - np.exp(-epoch * 0.2))
        else:
            # Validation metrics are slightly worse and noisier
            base_loss = 2.2 * np.exp(-epoch * 0.12) + 0.15
            base_accuracy = min(0.95, 0.95 - np.exp(-epoch * 0.18))
        
        # Add realistic noise
        loss = base_loss + np.random.normal(0, 0.05)
        accuracy = max(0.0, min(1.0, base_accuracy + np.random.normal(0, 0.02)))
        
        # Simulate step-by-step progress for training
        if is_training and steps_per_epoch > 50:  # Only for longer epochs
            for step in range(0, steps_per_epoch, max(1, steps_per_epoch // 10)):
                step_loss = loss + np.random.normal(0, 0.02)
                step_acc = accuracy + np.random.normal(0, 0.01)
                
                step_metrics = {
                    f'{phase}_loss_step': step_loss,
                    f'{phase}_accuracy_step': step_acc,
                    'step': global_step + step
                }
                
                if learning_rate:
                    step_metrics['learning_rate'] = learning_rate
                
                self.tracker.log_metrics(step_metrics, step=global_step + step)
                
                # Simulate step time
                time.sleep(0.01)
        
        # Simulate epoch computation time
        time.sleep(min(2.0, steps_per_epoch * 0.001))
        
        return {
            f'{phase}_loss': loss,
            f'{phase}_accuracy': accuracy,
            f'{phase}_perplexity': np.exp(loss),
            f'{phase}_f1_score': accuracy * 0.95 + np.random.normal(0, 0.01)
        }
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint_info = {
            'epoch': epoch,
            'model_state': f"checkpoint_epoch_{epoch}.pth",
            'optimizer_state': f"optimizer_epoch_{epoch}.pth",
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create dummy checkpoint file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(checkpoint_info, f, indent=2)
            checkpoint_file = f.name
        
        try:
            self.tracker.log_artifact(checkpoint_file, f"checkpoints/checkpoint_epoch_{epoch}.json")
            print(f"💾 Checkpoint saved for epoch {epoch}")
        finally:
            os.unlink(checkpoint_file)
    
    def _save_best_model(self, epoch: int, metrics: Dict[str, float]):
        """Save best model"""
        model_info = {
            'epoch': epoch,
            'metrics': metrics,
            'model_architecture': self.config['model']['architecture'],
            'hyperparameters': self.config['model']['hyperparameters'],
            'timestamp': datetime.now().isoformat(),
            'best_model': True
        }
        
        # Create dummy model file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(model_info, f, indent=2)
            model_file = f.name
        
        try:
            self.tracker.log_artifact(model_file, "models/best_model.json")
            
            # Register model in MLflow Model Registry
            model_name = f"{self.config['experiment']['name']}_model"
            self.tracker.register_model(
                model_name=model_name,
                model_path="models/best_model.json",
                description=f"Best model from epoch {epoch} with val_acc={metrics.get('val_accuracy', 0):.4f}"
            )
            
            print(f"🏆 Best model saved and registered: {model_name}")
        finally:
            os.unlink(model_file)
    
    def run_evaluation(self, best_accuracy: float):
        """Run comprehensive model evaluation"""
        print("\n🧪 Running model evaluation...")
        
        # Simulate evaluation on test set
        test_metrics = {
            'test_accuracy': best_accuracy * 0.98 + np.random.normal(0, 0.01),
            'test_loss': 0.15 + np.random.normal(0, 0.02),
            'test_f1_score': best_accuracy * 0.96 + np.random.normal(0, 0.01),
            'test_precision': best_accuracy * 0.97 + np.random.normal(0, 0.01),
            'test_recall': best_accuracy * 0.95 + np.random.normal(0, 0.01)
        }
        
        # Add domain-specific metrics
        test_metrics.update({
            'bleu_score': min(100, max(0, 85 + np.random.normal(0, 5))),
            'rouge_l': min(1.0, max(0, 0.8 + np.random.normal(0, 0.05))),
            'perplexity': np.exp(test_metrics['test_loss'])
        })
        
        # Log test metrics
        self.tracker.log_metrics(test_metrics)
        
        # Log to Prometheus
        if self.metrics_collector:
            self.metrics_collector.record_model_metrics(
                experiment_id=self.tracker.run_id,
                model_type=self.config['model']['architecture']['type'],
                accuracy=test_metrics['test_accuracy'],
                precision=test_metrics['test_precision'],
                recall=test_metrics['test_recall'],
                f1_score=test_metrics['test_f1_score'],
                dataset='test'
            )
        
        print("✅ Evaluation completed:")
        for metric, value in test_metrics.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
        
        return test_metrics
    
    def generate_final_report(self, test_metrics: Dict[str, float]):
        """Generate comprehensive experiment report"""
        print("\n📋 Generating final report...")
        
        # Collect all experiment data
        run_data = self.tracker.get_run_data()
        
        report = {
            'experiment_summary': {
                'name': self.tracker.experiment_name,
                'run_id': self.tracker.run_id,
                'start_time': run_data.get('start_time'),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - datetime.fromisoformat(run_data.get('start_time', datetime.now().isoformat()))).total_seconds() / 60
            },
            'configuration': self.config,
            'final_metrics': test_metrics,
            'best_validation_accuracy': max([m for k, m in run_data.get('metrics', {}).items() if 'val_accuracy' in k], default=0),
            'total_parameters': run_data.get('model_info', {}).get('total_parameters', 0),
            'artifacts': run_data.get('artifacts', []),
            'tags': run_data.get('tags', {}),
            'environment': run_data.get('environment', {})
        }
        
        # Save report as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(report, f, indent=2)
            report_file = f.name
        
        try:
            self.tracker.log_artifact(report_file, "reports/final_report.json")
            print("✅ Final report saved")
        finally:
            os.unlink(report_file)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Experiment: {report['experiment_summary']['name']}")
        print(f"Run ID: {report['experiment_summary']['run_id']}")
        print(f"Duration: {report['experiment_summary']['duration_minutes']:.1f} minutes")
        print(f"Best Val Accuracy: {report['best_validation_accuracy']:.4f}")
        print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        print(f"Model Parameters: {report['total_parameters']:,}")
        print(f"Artifacts: {len(report['artifacts'])} files")
        print("="*60)
        
        return report
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        try:
            # Finish the MLflow run
            self.tracker.finish_run(status="FINISHED")
            
            # Record experiment completion in Prometheus
            if self.metrics_collector:
                self.metrics_collector.record_experiment_completion(
                    model_type=self.config['model']['architecture']['type'],
                    success=True
                )
            
            print("✅ Experiment finished successfully")
            
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
            if self.tracker:
                self.tracker.finish_run(status="FAILED")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment configuration"""
    
    default_config = {
        'experiment': {
            'name': 'lezea_full_training_demo',
            'description': 'Complete LeZeA MLOps training demonstration with all features',
            'tags': {
                'team': 'ml-research',
                'priority': 'high',
                'framework': 'pytorch',
                'model_family': 'transformer'
            }
        },
        'model': {
            'architecture': {
                'type': 'transformer',
                'num_layers': 12,
                'hidden_size': 768,
                'num_heads': 12,
                'vocab_size': 50000,
                'max_sequence_length': 512,
                'dropout_rate': 0.1
            },
            'hyperparameters': {
                'learning_rate': 0.0001,
                'batch_size': 32,
                'weight_decay': 0.01,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8,
                'max_grad_norm': 1.0,
                'warmup_steps': 1000
            }
        },
        'data': {
            'dataset_name': 'lezea_training_data',
            'version': '1.2.0',
            'train_samples': 100000,
            'val_samples': 10000,
            'test_samples': 5000,
            'features': 768,
            'size_gb': 2.5,
            'preprocessing': ['tokenization', 'normalization', 'augmentation']
        },
        'training': {
            'epochs': 20,
            'optimizer': 'adamw',
            'scheduler': {
                'type': 'cosine',
                'warmup_epochs': 2,
                'min_lr': 1e-6
            },
            'regularization': {
                'dropout': 0.1,
                'weight_decay': 0.01,
                'label_smoothing': 0.1
            },
            'early_stopping': {
                'patience': 5,
                'min_delta': 0.001
            },
            'checkpoint_every': 3,
            'lr_decay_every': 8,
            'lr_decay_factor': 0.8
        }
    }
    
    if config_path and os.path.exists(config_path):
        print(f"📂 Loading config from {config_path}")
        with open(config_path) as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                custom_config = yaml.safe_load(f)
            else:
                custom_config = json.load(f)
        
        # Merge configurations (custom overrides default)
        def merge_configs(default, custom):
            for key, value in custom.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_configs(default[key], value)
                else:
                    default[key] = value
        
        merge_configs(default_config, custom_config)
    
    return default_config

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LeZeA MLOps Complete Training Example')
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--experiment-name', type=str, help='Override experiment name')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['model']['hyperparameters']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['model']['hyperparameters']['learning_rate'] = args.learning_rate
    
    print("🚀 LeZeA MLOps - Complete Training Example")
    print("="*60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Model: {config['model']['architecture']['type']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['model']['hyperparameters']['batch_size']}")
    print("="*60)
    
    # Initialize and run training
    trainer = LeZeATrainingExample(config)
    
    try:
        # Setup
        trainer.setup_experiment()
        trainer.log_configuration()
        
        # Data preparation
        dataset_info = trainer.simulate_data_loading()
        
        # Model initialization
        model_info = trainer.simulate_model_initialization()
        
        # Training loop
        best_accuracy = trainer.simulate_training_loop(dataset_info, model_info)
        
        # Evaluation
        test_metrics = trainer.run_evaluation(best_accuracy)
        
        # Final report
        report = trainer.generate_final_report(test_metrics)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"🌐 View results: http://localhost:5000")
        print(f"📊 Prometheus metrics: http://localhost:9090")
        print(f"📈 Grafana dashboards: http://localhost:3000")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if trainer.tracker:
            trainer.tracker.finish_run(status="FAILED")
        raise
    
    finally:
        # Always cleanup
        trainer.cleanup()

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
LeZeA MLOps - Complete Training Example
======================================

Comprehensive example showing all LeZeA MLOps features:
- Advanced experiment configuration
- Comprehensive logging and monitoring
- Resource tracking and optimization
- Model versioning and deployment
- Integration with all backends

This is the complete example Marcus can use as a template for production training.

Usage:
    python examples/full_training.py [--config config.yaml] [--experiment-name NAME]
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
from typing import Dict, List, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lezea_mlops import ExperimentTracker
from lezea_mlops.monitoring import get_metrics, GPUMonitor, EnvironmentTagger

class LeZeATrainingExample:
    """Complete LeZeA training example with all features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracker = None
        self.metrics_collector = None
        self.gpu_monitor = None
        
    def setup_experiment(self):
        """Initialize experiment tracking and monitoring"""
        print("🔧 Setting up experiment...")
        
        # Initialize experiment tracker with advanced configuration
        self.tracker = ExperimentTracker(
            experiment_name=self.config['experiment']['name'],
            description=self.config['experiment']['description'],
            tags=self.config['experiment'].get('tags', {}),
            create_if_not_exists=True
        )
        
        # Initialize metrics collection
        self.metrics_collector = get_metrics()
        
        # Initialize GPU monitoring if available
        try:
            self.gpu_monitor = GPUMonitor()
            print("✅ GPU monitoring initialized")
        except Exception as e:
            print(f"⚠️  GPU monitoring unavailable: {e}")
            self.gpu_monitor = None
        
        # Log environment information
        env_tagger = EnvironmentTagger()
        env_info = env_tagger.get_complete_environment()
        self.tracker.log_environment_info(env_info)
        
        print(f"✅ Experiment '{self.tracker.experiment_name}' initialized")
        print(f"📊 Run ID: {self.tracker.run_id}")
        print(f"🌐 MLflow UI: http://localhost:5000")
        
    def log_configuration(self):
        """Log complete experiment configuration"""
        print("📝 Logging experiment configuration...")
        
        # Log hyperparameters
        self.tracker.log_params(self.config['model']['hyperparameters'])
        
        # Log model architecture
        self.tracker.log_model_architecture(self.config['model']['architecture'])
        
        # Log dataset information
        self.tracker.log_dataset_info(self.config['data'])
        
        # Log training configuration
        training_config = {
            'optimizer': self.config['training']['optimizer'],
            'scheduler': self.config['training'].get('scheduler', {}),
            'regularization': self.config['training'].get('regularization', {}),
            'early_stopping': self.config['training'].get('early_stopping', {})
        }
        self.tracker.log_params(training_config)
        
        print("✅ Configuration logged")
    
    def simulate_data_loading(self) -> Dict[str, Any]:
        """Simulate data loading with monitoring"""
        print("📊 Loading and preparing data...")
        
        data_config = self.config['data']
        
        # Simulate data loading time
        load_time = np.random.uniform(2.0, 5.0)
        time.sleep(load_time)
        
        # Create simulated dataset metadata
        dataset_info = {
            'name': data_config['dataset_name'],
            'version': data_config['version'],
            'train_samples': data_config['train_samples'],
            'val_samples': data_config['val_samples'],
            'test_samples': data_config['test_samples'],
            'features': data_config['features'],
            'preprocessing': data_config['preprocessing'],
            'load_time_seconds': load_time
        }
        
        # Log dataset metrics
        self.tracker.log_dataset_metrics({
            'dataset_size_gb': data_config['size_gb'],
            'data_quality_score': 0.95,
            'missing_values_percent': 0.02,
            'data_loading_time': load_time
        })
        
        # Register dataset version with DVC
        try:
            self.tracker.dvc_backend.track_dataset(
                dataset_name=data_config['dataset_name'],
                version=data_config['version'],
                file_path=f"data/{data_config['dataset_name']}.parquet",
                metadata=dataset_info
            )
            print("✅ Dataset version tracked with DVC")
        except Exception as e:
            print(f"⚠️  DVC tracking failed: {e}")
        
        print(f"✅ Data loaded: {dataset_info['train_samples']:,} train, "
              f"{dataset_info['val_samples']:,} validation, "
              f"{dataset_info['test_samples']:,} test samples")
        
        return dataset_info
    
    def simulate_model_initialization(self) -> Dict[str, Any]:
        """Simulate model initialization with architecture logging"""
        print("🧠 Initializing model...")
        
        model_config = self.config['model']
        
        # Simulate model creation time
        init_time = np.random.uniform(1.0, 3.0)
        time.sleep(init_time)
        
        # Calculate model parameters
        architecture = model_config['architecture']
        total_params = (
            architecture['hidden_size'] * architecture['vocab_size'] +  # Embedding
            architecture['num_layers'] * architecture['hidden_size'] * architecture['hidden_size'] * 4 +  # Transformer layers
            architecture['hidden_size'] * architecture['vocab_size']  # Output layer
        )
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': int(total_params * 0.95),  # Some frozen params
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'initialization_time': init_time,
            'architecture_type': architecture['type']
        }
        
        # Log model information
        self.tracker.log_model_info(model_info)
        
        # Log to Prometheus if available
        if self.metrics_collector:
            self.metrics_collector.record_model_metrics(
                experiment_id=self.tracker.run_id,
                model_type=architecture['type']
            )
        
        print(f"✅ Model initialized: {total_params:,} parameters ({model_info['model_size_mb']:.1f} MB)")
        
        return model_info
    
    def simulate_training_loop(self, dataset_info: Dict[str, Any], model_info: Dict[str, Any]):
        """Simulate complete training loop with comprehensive monitoring"""
        print("🚀 Starting training loop...")
        
        training_config = self.config['training']
        num_epochs = training_config['epochs']
        batch_size = self.config['model']['hyperparameters']['batch_size']
        learning_rate = self.config['model']['hyperparameters']['learning_rate']
        
        # Calculate steps per epoch
        steps_per_epoch = dataset_info['train_samples'] // batch_size
        total_steps = num_epochs * steps_per_epoch
        
        print(f"📈 Training for {num_epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps")
        
        # Training state
        best_val_accuracy = 0.0
        no_improvement_count = 0
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            epoch_metrics = self._simulate_epoch(
                phase="train",
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                global_step=global_step,
                learning_rate=learning_rate
            )
            
            global_step += steps_per_epoch
            
            # Validation phase
            val_metrics = self._simulate_epoch(
                phase="validation",
                epoch=epoch,
                steps_per_epoch=dataset_info['val_samples'] // batch_size,
                global_step=global_step
            )
            
            # Combine metrics
            combined_metrics = {**epoch_metrics, **val_metrics}
            combined_metrics['epoch'] = epoch
            combined_metrics['learning_rate'] = learning_rate * (0.95 ** epoch)  # LR decay
            
            # Log epoch metrics
            self.tracker.log_metrics(combined_metrics, step=global_step)
            
            # Resource monitoring
            if self.gpu_monitor:
                gpu_stats = self.gpu_monitor.get_gpu_stats()
                if gpu_stats:
                    self.tracker.log_resource_usage({
                        'gpu_utilization': gpu_stats[0].get('utilization_percent', 0),
                        'gpu_memory_used_mb': gpu_stats[0].get('memory_used_mb', 0),
                        'gpu_temperature': gpu_stats[0].get('temperature', 0)
                    })
                    
                    # Log to Prometheus
                    if self.metrics_collector:
                        self.metrics_collector.update_gpu_metrics(gpu_stats)
            
            # Log to Prometheus
            if self.metrics_collector:
                epoch_time = time.time() - epoch_start_time
                self.metrics_collector.record_training_step(
                    experiment_id=self.tracker.run_id,
                    model_type=self.config['model']['architecture']['type'],
                    step_time=epoch_time / steps_per_epoch,
                    loss=epoch_metrics['train_loss'],
                    samples_per_sec=batch_size * steps_per_epoch / epoch_time
                )
                
                self.metrics_collector.record_epoch_completion(
                    experiment_id=self.tracker.run_id,
                    model_type=self.config['model']['architecture']['type'],
                    epoch_time=epoch_time
                )
            
            # Model checkpointing
            if epoch % training_config.get('checkpoint_every', 5) == 0:
                self._save_checkpoint(epoch, combined_metrics)
            
            # Early stopping check
            val_accuracy = val_metrics['val_accuracy']
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                no_improvement_count = 0
                self._save_best_model(epoch, combined_metrics)
            else:
                no_improvement_count += 1
            
            # Print epoch summary
            print(f"✅ Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.1f}s")
            print(f"   Train Loss: {epoch_metrics['train_loss']:.4f}, "
                  f"Train Acc: {epoch_metrics['train_accuracy']:.4f}")
            print(f"   Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.4f}")
            
            # Early stopping
            early_stop_patience = training_config.get('early_stopping', {}).get('patience', 10)
            if no_improvement_count >= early_stop_patience:
                print(f"🛑 Early stopping triggered after {no_improvement_count} epochs without improvement")
                break
            
            # Learning rate decay
            if epoch > 0 and epoch % training_config.get('lr_decay_every', 10) == 0:
                learning_rate *= training_config.get('lr_decay_factor', 0.8)
                print(f"📉 Learning rate decayed to {learning_rate:.6f}")
        
        print(f"🎯 Training completed! Best validation accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy
    
    def _simulate_epoch(self, phase: str, epoch: int, steps_per_epoch: int, 
                       global_step: int, learning_rate: float = None) -> Dict[str, float]:
        """Simulate a single epoch of training or validation"""
        
        is_training = phase == "train"
        
        # Simulate epoch with realistic loss curves
        if is_training:
            # Training loss decreases with some noise
            base_loss = 2.0 * np.exp(-epoch * 0.15) + 0.1
            base_accuracy = min(0.98, 1.0 - np.exp(-epoch * 0.2))
        else:
            # Validation metrics are slightly worse and noisier
            base_loss = 2.2 * np.exp(-epoch * 0.12) + 0.15
            base_accuracy = min(0.95, 0.95 - np.exp(-epoch * 0.18))
        
        # Add realistic noise
        loss = base_loss + np.random.normal(0, 0.05)
        accuracy = max(0.0, min(1.0, base_accuracy + np.random.normal(0, 0.02)))
        
        # Simulate step-by-step progress for training
        if is_training and steps_per_epoch > 50:  # Only for longer epochs
            for step in range(0, steps_per_epoch, max(1, steps_per_epoch // 10)):
                step_loss = loss + np.random.normal(0, 0.02)
                step_acc = accuracy + np.random.normal(0, 0.01)
                
                step_metrics = {
                    f'{phase}_loss_step': step_loss,
                    f'{phase}_accuracy_step': step_acc,
                    'step': global_step + step
                }
                
                if learning_rate:
                    step_metrics['learning_rate'] = learning_rate
                
                self.tracker.log_metrics(step_metrics, step=global_step + step)
                
                # Simulate step time
                time.sleep(0.01)
        
        # Simulate epoch computation time
        time.sleep(min(2.0, steps_per_epoch * 0.001))
        
        return {
            f'{phase}_loss': loss,
            f'{phase}_accuracy': accuracy,
            f'{phase}_perplexity': np.exp(loss),
            f'{phase}_f1_score': accuracy * 0.95 + np.random.normal(0, 0.01)
        }
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint_info = {
            'epoch': epoch,
            'model_state': f"checkpoint_epoch_{epoch}.pth",
            'optimizer_state': f"optimizer_epoch_{epoch}.pth",
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create dummy checkpoint file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(checkpoint_info, f, indent=2)
            checkpoint_file = f.name
        
        try:
            self.tracker.log_artifact(checkpoint_file, f"checkpoints/checkpoint_epoch_{epoch}.json")
            print(f"💾 Checkpoint saved for epoch {epoch}")
        finally:
            os.unlink(checkpoint_file)
    
    def _save_best_model(self, epoch: int, metrics: Dict[str, float]):
        """Save best model"""
        model_info = {
            'epoch': epoch,
            'metrics': metrics,
            'model_architecture': self.config['model']['architecture'],
            'hyperparameters': self.config['model']['hyperparameters'],
            'timestamp': datetime.now().isoformat(),
            'best_model': True
        }
        
        # Create dummy model file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(model_info, f, indent=2)
            model_file = f.name
        
        try:
            self.tracker.log_artifact(model_file, "models/best_model.json")
            
            # Register model in MLflow Model Registry
            model_name = f"{self.config['experiment']['name']}_model"
            self.tracker.register_model(
                model_name=model_name,
                model_path="models/best_model.json",
                description=f"Best model from epoch {epoch} with val_acc={metrics.get('val_accuracy', 0):.4f}"
            )
            
            print(f"🏆 Best model saved and registered: {model_name}")
        finally:
            os.unlink(model_file)
    
    def run_evaluation(self, best_accuracy: float):
        """Run comprehensive model evaluation"""
        print("\n🧪 Running model evaluation...")
        
        # Simulate evaluation on test set
        test_metrics = {
            'test_accuracy': best_accuracy * 0.98 + np.random.normal(0, 0.01),
            'test_loss': 0.15 + np.random.normal(0, 0.02),
            'test_f1_score': best_accuracy * 0.96 + np.random.normal(0, 0.01),
            'test_precision': best_accuracy * 0.97 + np.random.normal(0, 0.01),
            'test_recall': best_accuracy * 0.95 + np.random.normal(0, 0.01)
        }
        
        # Add domain-specific metrics
        test_metrics.update({
            'bleu_score': min(100, max(0, 85 + np.random.normal(0, 5))),
            'rouge_l': min(1.0, max(0, 0.8 + np.random.normal(0, 0.05))),
            'perplexity': np.exp(test_metrics['test_loss'])
        })
        
        # Log test metrics
        self.tracker.log_metrics(test_metrics)
        
        # Log to Prometheus
        if self.metrics_collector:
            self.metrics_collector.record_model_metrics(
                experiment_id=self.tracker.run_id,
                model_type=self.config['model']['architecture']['type'],
                accuracy=test_metrics['test_accuracy'],
                precision=test_metrics['test_precision'],
                recall=test_metrics['test_recall'],
                f1_score=test_metrics['test_f1_score'],
                dataset='test'
            )
        
        print("✅ Evaluation completed:")
        for metric, value in test_metrics.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
        
        return test_metrics
    
    def generate_final_report(self, test_metrics: Dict[str, float]):
        """Generate comprehensive experiment report"""
        print("\n📋 Generating final report...")
        
        # Collect all experiment data
        run_data = self.tracker.get_run_data()
        
        report = {
            'experiment_summary': {
                'name': self.tracker.experiment_name,
                'run_id': self.tracker.run_id,
                'start_time': run_data.get('start_time'),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - datetime.fromisoformat(run_data.get('start_time', datetime.now().isoformat()))).total_seconds() / 60
            },
            'configuration': self.config,
            'final_metrics': test_metrics,
            'best_validation_accuracy': max([m for k, m in run_data.get('metrics', {}).items() if 'val_accuracy' in k], default=0),
            'total_parameters': run_data.get('model_info', {}).get('total_parameters', 0),
            'artifacts': run_data.get('artifacts', []),
            'tags': run_data.get('tags', {}),
            'environment': run_data.get('environment', {})
        }
        
        # Save report as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(report, f, indent=2)
            report_file = f.name
        
        try:
            self.tracker.log_artifact(report_file, "reports/final_report.json")
            print("✅ Final report saved")
        finally:
            os.unlink(report_file)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Experiment: {report['experiment_summary']['name']}")
        print(f"Run ID: {report['experiment_summary']['run_id']}")
        print(f"Duration: {report['experiment_summary']['duration_minutes']:.1f} minutes")
        print(f"Best Val Accuracy: {report['best_validation_accuracy']:.4f}")
        print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        print(f"Model Parameters: {report['total_parameters']:,}")
        print(f"Artifacts: {len(report['artifacts'])} files")
        print("="*60)
        
        return report
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        try:
            # Finish the MLflow run
            self.tracker.finish_run(status="FINISHED")
            
            # Record experiment completion in Prometheus
            if self.metrics_collector:
                self.metrics_collector.record_experiment_completion(
                    model_type=self.config['model']['architecture']['type'],
                    success=True
                )
            
            print("✅ Experiment finished successfully")
            
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
            if self.tracker:
                self.tracker.finish_run(status="FAILED")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment configuration"""
    
    default_config = {
        'experiment': {
            'name': 'lezea_full_training_demo',
            'description': 'Complete LeZeA MLOps training demonstration with all features',
            'tags': {
                'team': 'ml-research',
                'priority': 'high',
                'framework': 'pytorch',
                'model_family': 'transformer'
            }
        },
        'model': {
            'architecture': {
                'type': 'transformer',
                'num_layers': 12,
                'hidden_size': 768,
                'num_heads': 12,
                'vocab_size': 50000,
                'max_sequence_length': 512,
                'dropout_rate': 0.1
            },
            'hyperparameters': {
                'learning_rate': 0.0001,
                'batch_size': 32,
                'weight_decay': 0.01,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8,
                'max_grad_norm': 1.0,
                'warmup_steps': 1000
            }
        },
        'data': {
            'dataset_name': 'lezea_training_data',
            'version': '1.2.0',
            'train_samples': 100000,
            'val_samples': 10000,
            'test_samples': 5000,
            'features': 768,
            'size_gb': 2.5,
            'preprocessing': ['tokenization', 'normalization', 'augmentation']
        },
        'training': {
            'epochs': 20,
            'optimizer': 'adamw',
            'scheduler': {
                'type': 'cosine',
                'warmup_epochs': 2,
                'min_lr': 1e-6
            },
            'regularization': {
                'dropout': 0.1,
                'weight_decay': 0.01,
                'label_smoothing': 0.1
            },
            'early_stopping': {
                'patience': 5,
                'min_delta': 0.001
            },
            'checkpoint_every': 3,
            'lr_decay_every': 8,
            'lr_decay_factor': 0.8
        }
    }
    
    if config_path and os.path.exists(config_path):
        print(f"📂 Loading config from {config_path}")
        with open(config_path) as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                custom_config = yaml.safe_load(f)
            else:
                custom_config = json.load(f)
        
        # Merge configurations (custom overrides default)
        def merge_configs(default, custom):
            for key, value in custom.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_configs(default[key], value)
                else:
                    default[key] = value
        
        merge_configs(default_config, custom_config)
    
    return default_config

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LeZeA MLOps Complete Training Example')
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--experiment-name', type=str, help='Override experiment name')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['model']['hyperparameters']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['model']['hyperparameters']['learning_rate'] = args.learning_rate
    
    print("🚀 LeZeA MLOps - Complete Training Example")
    print("="*60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Model: {config['model']['architecture']['type']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['model']['hyperparameters']['batch_size']}")
    print("="*60)
    
    # Initialize and run training
    trainer = LeZeATrainingExample(config)
    
    try:
        # Setup
        trainer.setup_experiment()
        trainer.log_configuration()
        
        # Data preparation
        dataset_info = trainer.simulate_data_loading()
        
        # Model initialization
        model_info = trainer.simulate_model_initialization()
        
        # Training loop
        best_accuracy = trainer.simulate_training_loop(dataset_info, model_info)
        
        # Evaluation
        test_metrics = trainer.run_evaluation(best_accuracy)
        
        # Final report
        report = trainer.generate_final_report(test_metrics)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"🌐 View results: http://localhost:5000")
        print(f"📊 Prometheus metrics: http://localhost:9090")
        print(f"📈 Grafana dashboards: http://localhost:3000")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if trainer.tracker:
            trainer.tracker.finish_run(status="FAILED")
        raise
    
    finally:
        # Always cleanup
        trainer.cleanup()

if __name__ == "__main__":
    main()#!/usr/bin/env python3