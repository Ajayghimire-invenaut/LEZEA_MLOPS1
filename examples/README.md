# LeZeA MLOps Examples

This directory contains practical examples showing how to use LeZeA MLOps for different scenarios. Each example is self-contained and demonstrates specific features and best practices.

## Quick Start Examples

### 1. quick_start.py - Minimal Example
**Perfect for first-time users and quick prototyping**

```bash
python examples/quick_start.py
```

**What it demonstrates:**
- Basic experiment initialization
- Parameter and metric logging
- Simple training loop
- Artifact storage
- Experiment completion

**Use this when:**
- Learning LeZeA MLOps for the first time
- Quick experimentation and prototyping
- Testing your setup
- Creating minimal reproducible examples

**Key features shown:**
- `ExperimentTracker` basic usage
- Parameter logging with `log_params()`
- Metric logging with `log_metrics()`
- Artifact storage with `log_artifact()`
- Experiment tags and metadata

### 2. full_training.py - Complete Production Example
**Comprehensive example showing all LeZeA MLOps features**

```bash
# Run with default configuration
python examples/full_training.py

# Run with custom configuration
python examples/full_training.py --config my_config.yaml

# Override specific parameters
python examples/full_training.py --experiment-name "my_experiment" --epochs 50 --batch-size 64
```

**What it demonstrates:**
- Advanced experiment configuration
- Comprehensive monitoring and logging
- Resource tracking (GPU, CPU, memory)
- Model versioning and registration
- Integration with all backends
- Production-ready patterns

**Use this when:**
- Setting up production training pipelines
- Need comprehensive monitoring
- Working with large-scale experiments
- Requiring full MLOps features
- Building team workflows

**Key features shown:**
- Configuration management (YAML/JSON)
- Environment and hardware profiling
- Real-time resource monitoring
- Step-by-step metric tracking
- Model checkpointing and versioning
- Automated artifact management
- Integration with Prometheus metrics
- Model registry usage
- Comprehensive reporting

## Configuration Examples

### Basic Configuration (quick_start.py)
```python
# Minimal setup
tracker = ExperimentTracker("my_experiment")

# Log parameters
tracker.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "model_type": "transformer"
})

# Log metrics during training
for epoch in range(10):
    metrics = {"loss": loss, "accuracy": accuracy}
    tracker.log_metrics(metrics, step=epoch)
```

### Advanced Configuration (full_training.py)
```yaml
# config.yaml
experiment:
  name: "production_experiment"
  description: "Production training with full monitoring"
  tags:
    team: "ml-research"
    priority: "high"
    framework: "pytorch"

model:
  architecture:
    type: "transformer"
    num_layers: 12
    hidden_size: 768
    vocab_size: 50000
  
  hyperparameters:
    learning_rate: 0.0001
    batch_size: 32
    weight_decay: 0.01

training:
  epochs: 20
  optimizer: "adamw"
  early_stopping:
    patience: 5
    min_delta: 0.001
```

## Usage Patterns

### Pattern 1: Research Experimentation
```python
# Quick iteration for research
from lezea_mlops import ExperimentTracker

tracker = ExperimentTracker("research_experiment")

# Try different hyperparameters
for lr in [0.001, 0.0001, 0.00001]:
    tracker.log_params({"learning_rate": lr})
    
    # Train model...
    final_loss = train_model(lr)
    
    tracker.log_metrics({"final_loss": final_loss})
    tracker.finish_run()
```

### Pattern 2: Production Training
```python
# Production training with full monitoring
from lezea_mlops import ExperimentTracker
from lezea_mlops.monitoring import get_metrics

# Initialize with comprehensive tracking
tracker = ExperimentTracker(
    experiment_name="production_model_v2",
    description="Production model training with A/B testing",
    tags={"environment": "production", "version": "2.0"}
)

# Enable metrics collection
metrics = get_metrics()
metrics.start_metrics_server()

# Training with monitoring
for epoch in range(epochs):
    # Training step
    loss = train_step()
    
    # Log to both MLflow and Prometheus
    tracker.log_metrics({"loss": loss}, step=epoch)
    metrics.record_training_step(
        experiment_id=tracker.run_id,
        model_type="transformer",
        step_time=step_time,
        loss=loss
    )
```

### Pattern 3: Hyperparameter Optimization
```python
# HPO with comprehensive tracking
import optuna
from lezea_mlops import ExperimentTracker

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Create experiment for this trial
    tracker = ExperimentTracker(f"hpo_trial_{trial.number}")
    tracker.log_params({"learning_rate": lr, "batch_size": batch_size})
    
    # Train and evaluate
    accuracy = train_and_evaluate(lr, batch_size)
    tracker.log_metrics({"accuracy": accuracy})
    tracker.finish_run()
    
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

### Pattern 4: Model Comparison
```python
# Compare different model architectures
models = ["transformer", "cnn", "rnn"]
results = {}

for model_type in models:
    tracker = ExperimentTracker(f"model_comparison_{model_type}")
    
    # Log model-specific parameters
    tracker.log_params({"model_type": model_type})
    tracker.set_tags({"comparison": "architecture_study"})
    
    # Train model
    metrics = train_model(model_type)
    tracker.log_metrics(metrics)
    
    # Store results
    results[model_type] = metrics["test_accuracy"]
    tracker.finish_run()

# Find best model
best_model = max(results, key=results.get)
print(f"Best model: {best_model} with accuracy {results[best_model]}")
```

## Environment Setup

### Prerequisites
```bash
# Ensure all services are running
./scripts/start_all.sh

# Verify health
python scripts/health_check.py
```

### Environment Variables
```bash
# .env file for examples
MLFLOW_TRACKING_URI=http://localhost:5000
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=lezea_mlops
POSTGRES_USER=lezea_user
POSTGRES_PASSWORD=lezea_secure_password_2024

# S3/MinIO configuration
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://lezea-mlops-artifacts

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
```

## Running the Examples

### 1. Start Services
```bash
# Start all LeZeA MLOps services
./scripts/start_all.sh

# Verify everything is running
./scripts/health_check.py
```

### 2. Run Quick Start
```bash
cd lezea-mlops
python examples/quick_start.py
```

**Expected output:**
```
🚀 LeZeA MLOps Quick Start Example
==================================================
✅ Experiment initialized: quick_start_demo
📊 Run ID: abc123...
✅ Parameters logged
✅ Tags added

🔄 Starting simulated training...
Epoch  1/10: Loss=1.8234, Accuracy=0.4521
Epoch  2/10: Loss=1.2341, Accuracy=0.6234
...
✅ Model metadata logged
💾 Logging artifacts...
✅ Model artifact logged

📈 Experiment Summary:
------------------------------
train_loss: 0.1234
train_accuracy: 0.9567
learning_rate: 0.0008

🎯 Experiment completed successfully!
📊 View in MLflow UI: http://localhost:5000
🔍 Run ID: abc123...
✅ Experiment finished and logged
```

### 3. Run Full Training Example
```bash
# Basic run
python examples/full_training.py

# With custom configuration
python examples/full_training.py --config examples/config.yaml --epochs 30

# With parameter overrides
python examples/full_training.py --experiment-name "my_production_model" --batch-size 64 --learning-rate 0.0005
```

**Expected output:**
```
🚀 LeZeA MLOps - Complete Training Example
============================================================
Experiment: lezea_full_training_demo
Model: transformer
Epochs: 20
Batch Size: 32
============================================================

🔧 Setting up experiment...
✅ GPU monitoring initialized
✅ Experiment 'lezea_full_training_demo' initialized
📊 Run ID: def456...
🌐 MLflow UI: http://localhost:5000

📝 Logging experiment configuration...
✅ Configuration logged

📊 Loading and preparing data...
✅ Dataset version tracked with DVC
✅ Data loaded: 100,000 train, 10,000 validation, 5,000 test samples

🧠 Initializing model...
✅ Model initialized: 124,440,000 parameters (474.8 MB)

🚀 Starting training loop...
📈 Training for 20 epochs, 3125 steps/epoch, 62500 total steps

🔄 Epoch 1/20
✅ Epoch 1 completed in 32.1s
   Train Loss: 1.8234, Train Acc: 0.4521
   Val Loss: 1.9876, Val Acc: 0.4123
💾 Checkpoint saved for epoch 0
🏆 Best model saved and registered: lezea_full_training_demo_model

...

🧪 Running model evaluation...
✅ Evaluation completed:
   test_accuracy: 0.9234
   test_loss: 0.1456
   test_f1_score: 0.9156
   bleu_score: 87.34
   rouge_l: 0.8234

📋 Generating final report...
✅ Final report saved

============================================================
🎯 EXPERIMENT SUMMARY
============================================================
Experiment: lezea_full_training_demo
Run ID: def456...
Duration: 12.3 minutes
Best Val Accuracy: 0.9456
Test Accuracy: 0.9234
Model Parameters: 124,440,000
Artifacts: 8 files
============================================================

🎉 Training completed successfully!
🌐 View results: http://localhost:5000
📊 Prometheus metrics: http://localhost:9090
📈 Grafana dashboards: http://localhost:3000

🧹 Cleaning up...
✅ Experiment finished successfully
```

## Customization Guide

### Creating Your Own Example

1. **Start with the template:**
```python
#!/usr/bin/env python3
"""
My Custom LeZeA MLOps Example
============================
"""

from lezea_mlops import ExperimentTracker

def main():
    # Initialize tracker
    tracker = ExperimentTracker("my_custom_experiment")
    
    try:
        # Your training code here
        pass
        
    finally:
        # Always finish the run
        tracker.finish_run()

if __name__ == "__main__":
    main()
```

2. **Add your specific functionality:**
   - Data loading and preprocessing
   - Model architecture definition
   - Training loop implementation
   - Evaluation metrics
   - Custom artifact logging

3. **Use LeZeA MLOps features:**
   - Parameter logging: `tracker.log_params()`
   - Metric tracking: `tracker.log_metrics()`
   - Artifact storage: `tracker.log_artifact()`
   - Model registration: `tracker.register_model()`
   - Resource monitoring: `tracker.log_resource_usage()`

### Configuration Files

Create `my_config.yaml`:
```yaml
experiment:
  name: "my_custom_experiment"
  description: "My custom training experiment"
  tags:
    author: "your_name"
    project: "my_project"

model:
  type: "custom_model"
  parameters:
    learning_rate: 0.001
    batch_size: 32

data:
  dataset: "my_dataset"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

training:
  epochs: 50
  optimizer: "adam"
  loss_function: "cross_entropy"
```

Load in your script:
```python
import yaml

with open("my_config.yaml") as f:
    config = yaml.safe_load(f)

tracker = ExperimentTracker(config['experiment']['name'])
tracker.log_params(config['model']['parameters'])
```

## Integration Examples

### With PyTorch
```python
import torch
import torch.nn as nn
from lezea_mlops import ExperimentTracker

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Define your model architecture
    
    def forward(self, x):
        # Model forward pass
        return x

def train_pytorch_model():
    tracker = ExperimentTracker("pytorch_experiment")
    
    # Log model configuration
    config = {"hidden_size": 256, "num_layers": 3}
    tracker.log_params(config)
    
    # Create model
    model = MyModel(config)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    tracker.log_model_info({"total_parameters": total_params})
    
    # Training loop
    for epoch in range(10):
        # Training step
        loss = train_step(model, optimizer)
        
        # Log metrics
        tracker.log_metrics({"loss": loss.item()}, step=epoch)
    
    # Save model
    torch.save(model.state_dict(), "model.pth")
    tracker.log_artifact("model.pth")
    
    tracker.finish_run()
```

### With Hugging Face Transformers
```python
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from lezea_mlops import ExperimentTracker

class LeZeAMLOpsCallback:
    def __init__(self, tracker):
        self.tracker = tracker
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            self.tracker.log_metrics(logs, step=state.global_step)

def train_with_transformers():
    tracker = ExperimentTracker("transformers_experiment")
    
    # Model setup
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_steps=10,
    )
    
    # Log parameters
    tracker.log_params(training_args.to_dict())
    
    # Create trainer with LeZeA MLOps callback
    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[LeZeAMLOpsCallback(tracker)]
    )
    
    # Train
    trainer.train()
    
    # Save and log model
    trainer.save_model("./final_model")
    tracker.log_artifact("./final_model")
    
    tracker.finish_run()
```

### With Scikit-learn
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from lezea_mlops import ExperimentTracker
import joblib

def train_sklearn_model():
    tracker = ExperimentTracker("sklearn_experiment")
    
    # Model parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    tracker.log_params(params)
    
    # Create and train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    tracker.log_metrics({
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std()
    })
    
    # Evaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    tracker.log_metrics({
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1_score": report["macro avg"]["f1-score"]
    })
    
    # Save model
    joblib.dump(model, "model.pkl")
    tracker.log_artifact("model.pkl")
    
    tracker.finish_run()
```

## Troubleshooting

### Common Issues

1. **Services not running:**
```bash
# Check service status
./scripts/start_all.sh status

# Start missing services
./scripts/start_all.sh start
```

2. **Import errors:**
```bash
# Ensure you're in the project root
cd /path/to/lezea-mlops

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

3. **Permission errors:**
```bash
# Fix script permissions
chmod +x examples/*.py
chmod +x scripts/*.sh
```

4. **Connection errors:**
```bash
# Check connectivity
python scripts/health_check.py

# Check specific services
curl http://localhost:5000/health  # MLflow
curl http://localhost:9090/-/healthy  # Prometheus
```

### Getting Help

- **View experiment logs:** Check MLflow UI at http://localhost:5000
- **Monitor system health:** Use health check script: `python scripts/health_check.py`
- **Check service logs:** Use `./scripts/start_all.sh logs`
- **Prometheus metrics:** Visit http://localhost:9090
- **Grafana dashboards:** Visit http://localhost:3000

## Next Steps

1. **Start with quick_start.py** to familiarize yourself with basic concepts
2. **Explore full_training.py** to see advanced features
3. **Create your own example** using the patterns shown here
4. **Integrate with your existing ML code** using the framework examples
5. **Set up production workflows** based on the full training example

For more detailed documentation, see the main LeZeA MLOps README and individual component documentation.