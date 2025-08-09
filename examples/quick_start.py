#!/usr/bin/env python3
"""
LeZeA MLOps - Quick Start Example
================================

This is the minimal example Marcus can copy-paste to get started with LeZeA MLOps.
Demonstrates core functionality in under 50 lines of code.

What this example shows:
- Basic experiment tracking setup
- Parameter and metric logging
- Model artifact storage
- Simple training loop with monitoring

Usage:
    python examples/quick_start.py
"""

import time
import random
import numpy as np
from lezea_mlops import ExperimentTracker

def main():
    """Quick start example - minimal LeZeA MLOps usage"""
    
    print("🚀 LeZeA MLOps Quick Start Example")
    print("="*50)
    
    # 1. Initialize experiment tracker
    # This automatically connects to all configured backends
    tracker = ExperimentTracker(
        experiment_name="quick_start_demo",
        description="Minimal example of LeZeA MLOps functionality"
    )
    
    print(f"✅ Experiment initialized: {tracker.experiment_name}")
    print(f"📊 Run ID: {tracker.run_id}")
    
    # 2. Log hyperparameters
    params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "model_type": "transformer",
        "dataset": "sample_data"
    }
    
    tracker.log_params(params)
    print("✅ Parameters logged")
    
    # 3. Add experiment tags
    tracker.set_tags({
        "team": "ml-research",
        "priority": "high",
        "framework": "pytorch"
    })
    print("✅ Tags added")
    
    # 4. Simulate training loop with metrics
    print("\n🔄 Starting simulated training...")
    
    for epoch in range(params["epochs"]):
        # Simulate training step
        time.sleep(0.5)  # Simulate computation time
        
        # Simulate improving metrics with some noise
        base_loss = 2.0 * np.exp(-epoch * 0.3)
        loss = base_loss + random.uniform(-0.1, 0.1)
        
        base_accuracy = 1.0 - np.exp(-epoch * 0.4)
        accuracy = min(0.99, base_accuracy + random.uniform(-0.05, 0.05))
        
        # Log metrics for this epoch
        metrics = {
            "train_loss": loss,
            "train_accuracy": accuracy,
            "learning_rate": params["learning_rate"] * (0.95 ** epoch),  # Learning rate decay
            "epoch": epoch
        }
        
        tracker.log_metrics(metrics, step=epoch)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{params['epochs']}: "
              f"Loss={loss:.4f}, Accuracy={accuracy:.4f}")
    
    # 5. Log final model metadata
    model_info = {
        "model_size_mb": 125.7,
        "parameters_count": 12_500_000,
        "final_accuracy": accuracy,
        "training_time_minutes": params["epochs"] * 0.5 / 60
    }
    
    tracker.log_model_info(model_info)
    print("✅ Model metadata logged")
    
    # 6. Log some artifacts (simulate model files)
    print("\n💾 Logging artifacts...")
    
    # Create dummy model file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(f"Dummy model file\nAccuracy: {accuracy:.4f}\nParameters: {model_info['parameters_count']}")
        model_file = f.name
    
    try:
        tracker.log_artifact(model_file, "model.txt")
        print("✅ Model artifact logged")
    finally:
        os.unlink(model_file)  # Clean up temp file
    
    # 7. Get experiment summary
    print("\n📈 Experiment Summary:")
    print("-" * 30)
    
    # Get final metrics
    final_metrics = tracker.get_latest_metrics()
    for metric_name, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
    
    print(f"\n🎯 Experiment completed successfully!")
    print(f"📊 View in MLflow UI: http://localhost:5000")
    print(f"🔍 Run ID: {tracker.run_id}")
    
    # 8. Finish the experiment
    tracker.finish_run(status="FINISHED")
    print("✅ Experiment finished and logged")

if __name__ == "__main__":
    main()