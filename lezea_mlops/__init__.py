"""
LeZeA MLOps - Complete Experiment Tracking System
=================================================

A comprehensive MLOps platform designed specifically for LeZeA AGI experiments.
Provides unified tracking for experiments, models, datasets, resources, and business metrics.

Key Features:
- 🎯 Unified experiment tracking (MLflow integration)
- 📊 Complex data storage (MongoDB for modification trees, resource stats)
- ☁️ Scalable artifact storage (S3 for checkpoints, models)
- 📈 Real-time monitoring (Prometheus + Grafana)
- 📦 Dataset versioning (DVC integration)
- 🔧 Automated environment detection
- 💰 Business metrics and cost tracking

Quick Start:
-----------
```python
from lezea_mlops import ExperimentTracker

# Context manager (recommended)
with ExperimentTracker("my_experiment", "Testing LeZeA") as tracker:
    # Configure LeZeA
    tracker.log_lezea_config(
        tasker_pop_size=100,
        builder_pop_size=50,
        algorithm_type="SAC"
    )
    
    # Track training
    for step in range(100):
        tracker.log_training_step(step, loss=0.5, accuracy=0.8)
    
    # Save checkpoint
    tracker.log_checkpoint("model.pth", step=100)

# Manual usage
tracker = ExperimentTracker("my_experiment")
tracker.start()
# ... training code ...
tracker.end()
```

Architecture:
- tracker.py: Main interface (what Marcus uses)
- backends/: Storage system integrations (MLflow, MongoDB, S3, PostgreSQL, DVC)
- monitoring/: Real-time metrics (GPU, CPU, Prometheus)
- config/: Configuration management (YAML + environment variables)
- utils/: Helper utilities (logging, validation)

For more examples, see the examples/ directory.
"""

import sys
import warnings
from pathlib import Path
import logging

# Version information
__version__ = "1.0.0"
__author__ = "LeZeA Team"
__email__ = "mlops@lezea.ai"
__description__ = "Complete MLOps platform for LeZeA AGI experiments"

# Minimum Python version check
MIN_PYTHON_VERSION = (3, 8)
if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError(
        f"LeZeA MLOps requires Python {'.'.join(map(str, MIN_PYTHON_VERSION))} or higher. "
        f"You are running Python {'.'.join(map(str, sys.version_info[:2]))}."
    )

# Configuration validation on import
try:
    from .config import config

    # Validate critical configurations
    validation_results = config.validate_config()
    failed_services = [service for service, valid in validation_results.items() if not valid]

    if failed_services:
        warnings.warn(
            f"Some services are not properly configured: {', '.join(failed_services)}. "
            f"Check your .env file and service configurations. "
            f"The system will work with limited functionality.",
            UserWarning,
            stacklevel=2,
        )

    # Print configuration summary if in development mode
    if any(arg in sys.argv for arg in ["--dev", "--debug", "--verbose"]):
        config.print_config_summary()
except Exception as e:
    warnings.warn(
        f"Failed to load configuration: {e}. "
        f"Some features may not work properly.",
        UserWarning,
        stacklevel=2,
    )
    # Provide a minimal shim so later helpers don't crash
    class _DummyConfig:
        def validate_config(self):
            return {}
    config = locals().get("config", _DummyConfig())

# Main imports - what users interact with
_TRACKER_AVAILABLE = False
_TRACKER_IMPORT_ERROR = None

try:
    from .tracker import ExperimentTracker  # the real one
    _TRACKER_AVAILABLE = True
except Exception as err:
    # Surface the real reason with full traceback
    import traceback
    _TRACKER_AVAILABLE = False
    _TRACKER_IMPORT_ERROR = err
    print("❌ ExperimentTracker import failed with the real error:\n")
    traceback.print_exc()

    # Graceful degradation: provide a dummy that raises with original cause
    class ExperimentTracker:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ExperimentTracker import failed. See traceback above."
            ) from _TRACKER_IMPORT_ERROR

# Backend availability check
_BACKENDS_STATUS = {}

def check_backend_availability():
    """Check which backends are available"""
    backends = {
        "mlflow": ("mlflow", "MLflow experiment tracking"),
        "mongodb": ("pymongo", "MongoDB complex data storage"),
        "s3": ("boto3", "AWS S3 artifact storage"),
        "postgres": ("psycopg2", "PostgreSQL metadata storage"),
        "dvc": ("dvc", "DVC dataset versioning"),
        "monitoring": ("psutil", "System monitoring"),
    }

    for backend_name, (module_name, description) in backends.items():
        try:
            __import__(module_name)
            _BACKENDS_STATUS[backend_name] = True
        except ImportError:
            _BACKENDS_STATUS[backend_name] = False
            if any(arg in sys.argv for arg in ["--dev", "--debug", "--verbose"]):
                print(f"⚠️ {description} not available (missing {module_name})")

# Check backends on import
check_backend_availability()

# Utility functions for users
def get_version():
    """Get the current version of LeZeA MLOps"""
    return __version__

def get_backend_status():
    """Get the status of all backends"""
    return _BACKENDS_STATUS.copy()

def check_system_health():
    """
    Perform a basic system health check

    Returns:
        dict: Health status of all components
    """
    health_status = {
        "tracker_available": _TRACKER_AVAILABLE,
        "backends": get_backend_status(),
        "config_valid": True,
    }

    try:
        # Check configuration validity
        validation_results = config.validate_config()
        health_status["config_services"] = validation_results
        health_status["config_valid"] = all(validation_results.values())
    except Exception as e:
        health_status["config_valid"] = False
        health_status["config_error"] = str(e)

    return health_status

def print_system_info():
    """Print comprehensive system information"""
    print(f"\n🚀 LeZeA MLOps v{__version__}")
    print("=" * 50)

    # Python and system info
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")

    # Backend status
    print("\n📦 Backend Status:")
    for backend, available in _BACKENDS_STATUS.items():
        status = "✅" if available else "❌"
        print(f"  {status} {backend}")

    # Configuration status
    try:
        validation_results = config.validate_config()
        print("\n⚙️ Service Configuration:")
        for service, valid in validation_results.items():
            status = "✅" if valid else "❌"
            print(f"  {status} {service}")
    except Exception as e:
        print(f"\n❌ Configuration Error: {e}")

    print("=" * 50)

# Context manager for temporary configuration changes
class ConfigContext:
    """Context manager for temporary configuration overrides"""

    def __init__(self, **overrides):
        self.overrides = overrides
        self.original_values = {}

    def __enter__(self):
        # Store original values and apply overrides
        for key, value in self.overrides.items():
            if hasattr(config, key):
                self.original_values[key] = getattr(config, key)
                setattr(config, key, value)
        return config

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        for key, value in self.original_values.items():
            setattr(config, key, value)

# Development and debugging utilities
def enable_debug_mode():
    """Enable debug mode with verbose logging"""
    logging.basicConfig(level=logging.DEBUG)
    print("🐛 Debug mode enabled")

def create_sample_experiment():
    """Create a sample experiment for testing"""
    if not _TRACKER_AVAILABLE:
        print("❌ ExperimentTracker not available")
        return None

    print("🧪 Creating sample experiment...")
    tracker = ExperimentTracker("sample_experiment", "Testing LeZeA MLOps system")
    return tracker

# Export public API
__all__ = [
    # Main interface
    "ExperimentTracker",
    # Version and info
    "__version__",
    "get_version",
    "get_backend_status",
    # Health and debugging
    "check_system_health",
    "print_system_info",
    "enable_debug_mode",
    # Configuration
    "config",
    "ConfigContext",
    # Development utilities
    "create_sample_experiment",
]

# Auto-configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Print welcome message if in interactive mode
if hasattr(sys, "ps1") or sys.flags.interactive:
    print(f"🚀 LeZeA MLOps v{__version__} loaded")
    print(" Use ExperimentTracker for experiment tracking")
    print(" Run print_system_info() for system status")

# Cleanup
del sys, warnings, Path