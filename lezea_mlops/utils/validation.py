"""
Validation utilities for LeZeA MLOps
===================================

Input validation, data type checking, and constraint enforcement for all MLOps operations.
Provides comprehensive validation for experiments, models, data, and configurations.
"""

import re
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging
from math import isfinite

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass


class LeZeAValidator:
    """
    Comprehensive validation system for LeZeA MLOps

    Validates:
    - Experiment names and IDs
    - Model parameters and architectures
    - File paths and data formats
    - Service configurations
    - Performance metrics
    """

    # Valid patterns
    EXPERIMENT_NAME_PATTERN = r'^[a-zA-Z0-9_-]+$'
    MODEL_TYPE_PATTERN = r'^[a-zA-Z0-9_]+$'
    TAG_PATTERN = r'^[a-zA-Z0-9_.-]+$'

    # Size limits
    MAX_EXPERIMENT_NAME_LENGTH = 100
    MAX_DESCRIPTION_LENGTH = 1000
    MAX_TAG_LENGTH = 50
    MAX_TAGS_COUNT = 20
    MAX_METRIC_NAME_LENGTH = 50
    MAX_PARAM_NAME_LENGTH = 50

    # File size limits (in bytes)
    MAX_CONFIG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_LOG_FILE_SIZE = 100 * 1024 * 1024    # 100MB

    @classmethod
    def validate_experiment_name(cls, name: str) -> bool:
        """
        Validate experiment name format and constraints

        Args:
            name: Experiment name to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(name, str):
            raise ValidationError(f"Experiment name must be string, got {type(name)}")

        if not name.strip():
            raise ValidationError("Experiment name cannot be empty")

        if len(name) > cls.MAX_EXPERIMENT_NAME_LENGTH:
            raise ValidationError(
                f"Experiment name too long: {len(name)} > {cls.MAX_EXPERIMENT_NAME_LENGTH}"
            )

        if not re.match(cls.EXPERIMENT_NAME_PATTERN, name):
            raise ValidationError(
                f"Invalid experiment name format: {name}. Use only letters, numbers, underscore, hyphen"
            )

        # Reserved names
        reserved = ['admin', 'system', 'default', 'test', 'debug', 'temp', 'tmp']
        if name.lower() in reserved:
            raise ValidationError(f"Experiment name '{name}' is reserved")

        return True

    @classmethod
    def validate_model_params(cls, params: Dict[str, Any]) -> bool:
        """
        Validate model parameters dictionary

        Args:
            params: Dictionary of model parameters

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(params, dict):
            raise ValidationError(f"Model parameters must be dict, got {type(params)}")

        for key, value in params.items():
            # Validate parameter name
            if not isinstance(key, str):
                raise ValidationError(f"Parameter name must be string, got {type(key)}: {key}")

            if len(key) > cls.MAX_PARAM_NAME_LENGTH:
                raise ValidationError(
                    f"Parameter name too long: {len(key)} > {cls.MAX_PARAM_NAME_LENGTH}"
                )

            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', key):
                raise ValidationError(f"Invalid parameter name format: {key}")

            # Validate parameter value
            cls._validate_param_value(key, value)

        return True

    @classmethod
    def _validate_param_value(cls, name: str, value: Any) -> None:
        """Validate individual parameter value"""
        # Check for valid types
        valid_types = (int, float, str, bool, list, dict, type(None))
        if not isinstance(value, valid_types):
            raise ValidationError(f"Invalid parameter type for '{name}': {type(value)}")

        # String length limits
        if isinstance(value, str) and len(value) > 1000:
            raise ValidationError(f"Parameter '{name}' string value too long: {len(value)} > 1000")

        # Numeric ranges for common LeZeA parameters
        numeric_ranges = {
            'learning_rate': (1e-6, 1.0),
            'batch_size': (1, 10000),
            'epochs': (1, 10000),
            'dropout_rate': (0.0, 1.0),
            'weight_decay': (0.0, 1.0),
            'momentum': (0.0, 1.0),
            'beta1': (0.0, 1.0),
            'beta2': (0.0, 1.0),
            'epsilon': (1e-12, 1e-3),
            'max_grad_norm': (0.1, 100.0),
            'warmup_steps': (0, 100000),
            'num_layers': (1, 100),
            'hidden_size': (8, 8192),
            'num_heads': (1, 64),
            'vocab_size': (1, 1000000),
            'seq_length': (1, 100000),
        }

        if isinstance(value, (int, float)) and name in numeric_ranges:
            min_val, max_val = numeric_ranges[name]
            if not (min_val <= value <= max_val):
                raise ValidationError(
                    f"Parameter '{name}' value {value} outside valid range [{min_val}, {max_val}]"
                )

    @classmethod
    def validate_metrics(cls, metrics: Dict[str, Union[float, int]]) -> bool:
        """
        Validate metrics dictionary

        Args:
            metrics: Dictionary of metric name -> value

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(metrics, dict):
            raise ValidationError(f"Metrics must be dict, got {type(metrics)}")

        for name, value in metrics.items():
            # Validate metric name
            if not isinstance(name, str):
                raise ValidationError(f"Metric name must be string, got {type(name)}: {name}")

            if len(name) > cls.MAX_METRIC_NAME_LENGTH:
                raise ValidationError(
                    f"Metric name too long: {len(name)} > {cls.MAX_METRIC_NAME_LENGTH}"
                )

            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_./-]*$', name):
                raise ValidationError(f"Invalid metric name format: {name}")

            # Validate metric value
            if not isinstance(value, (int, float)):
                raise ValidationError(
                    f"Metric '{name}' value must be numeric, got {type(value)}: {value}"
                )

            if not isfinite(float(value)):
                raise ValidationError(f"Metric '{name}' is NaN or infinite")

            if not (-1e10 <= float(value) <= 1e10):
                raise ValidationError(f"Metric '{name}' value {value} outside reasonable range")

        return True

    @classmethod
    def validate_tags(cls, tags: List[str]) -> bool:
        """
        Validate experiment tags

        Args:
            tags: List of tag strings

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(tags, list):
            raise ValidationError(f"Tags must be list, got {type(tags)}")

        if len(tags) > cls.MAX_TAGS_COUNT:
            raise ValidationError(f"Too many tags: {len(tags)} > {cls.MAX_TAGS_COUNT}")

        for tag in tags:
            if not isinstance(tag, str):
                raise ValidationError(f"Tag must be string, got {type(tag)}: {tag}")

            if not tag.strip():
                raise ValidationError("Tag cannot be empty")

            if len(tag) > cls.MAX_TAG_LENGTH:
                raise ValidationError(f"Tag too long: {len(tag)} > {cls.MAX_TAG_LENGTH}")

            if not re.match(cls.TAG_PATTERN, tag):
                raise ValidationError(f"Invalid tag format: {tag}")

        # Check for duplicates
        if len(set(tags)) != len(tags):
            raise ValidationError("Duplicate tags found")

        return True

    @classmethod
    def validate_file_path(
        cls,
        path: Union[str, Path],
        must_exist: bool = True,
        check_permissions: bool = True
    ) -> bool:
        """
        Validate file path and accessibility

        Args:
            path: File path to validate
            must_exist: Whether file must already exist
            check_permissions: Whether to check read/write permissions

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise ValidationError(f"Path must be string or Path, got {type(path)}")

        # Basic path validation
        if not str(path).strip():
            raise ValidationError("Path cannot be empty")

        # Check for dangerous path patterns
        path_str = str(path)
        if any(danger in path_str for danger in ['../', '~/', '/etc/', '/root/', '/sys/', '/proc/']):
            raise ValidationError(f"Potentially dangerous path: {path}")

        # Existence check
        if must_exist and not path.exists():
            raise ValidationError(f"Path does not exist: {path}")

        if path.exists():
            # Permission checks
            if check_permissions:
                if not os.access(path, os.R_OK):
                    raise ValidationError(f"No read permission for: {path}")

                # Check parent directory write permission for new files
                parent = path.parent
                if not os.access(parent, os.WORK_OK if hasattr(os, "WORK_OK") else os.W_OK):
                    raise ValidationError(f"No write permission for directory: {parent}")

            # File size checks
            if path.is_file():
                size = path.stat().st_size

                # Different limits for different file types
                if path.suffix in ['.yml', '.yaml', '.json']:
                    if size > cls.MAX_CONFIG_FILE_SIZE:
                        raise ValidationError(
                            f"Config file too large: {size} > {cls.MAX_CONFIG_FILE_SIZE}"
                        )

                elif path.suffix == '.log':
                    if size > cls.MAX_LOG_FILE_SIZE:
                        raise ValidationError(
                            f"Log file too large: {size} > {cls.MAX_LOG_FILE_SIZE}"
                        )

        return True

    @classmethod
    def validate_config_file(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate and load configuration file

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded configuration dictionary

        Raises:
            ValidationError: If validation fails
        """
        path = Path(config_path)
        cls.validate_file_path(path, must_exist=True)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix in ['.yml', '.yaml']:
                    config = yaml.safe_load(f)
                elif path.suffix == '.json':
                    config = json.load(f)
                else:
                    raise ValidationError(f"Unsupported config file format: {path.suffix}")
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValidationError(f"Invalid config file format: {e}")
        except Exception as e:
            raise ValidationError(f"Error reading config file: {e}")

        if not isinstance(config, dict):
            raise ValidationError("Config file must contain a dictionary")

        return config

    @classmethod
    def validate_dataset_info(cls, dataset_info: Dict[str, Any]) -> bool:
        """
        Validate dataset information dictionary

        Args:
            dataset_info: Dataset metadata dictionary

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(dataset_info, dict):
            raise ValidationError(f"Dataset info must be dict, got {type(dataset_info)}")

        # Required fields
        required_fields = ['name', 'version', 'size', 'format']
        for field in required_fields:
            if field not in dataset_info:
                raise ValidationError(f"Missing required dataset field: {field}")

        # Validate name
        name = dataset_info['name']
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Dataset name must be non-empty string")

        # Validate version
        version = dataset_info['version']
        if not isinstance(version, str) or not re.match(r'^\d+\.\d+\.\d+$', version):
            raise ValidationError(
                f"Dataset version must be semantic version (X.Y.Z), got: {version}"
            )

        # Validate size
        size = dataset_info['size']
        if not isinstance(size, int) or size < 0:
            raise ValidationError(f"Dataset size must be non-negative integer, got: {size}")

        # Validate format
        format_type = dataset_info['format']
        valid_formats = ['csv', 'json', 'parquet', 'pickle', 'hdf5', 'txt', 'binary']
        if format_type not in valid_formats:
            raise ValidationError(
                f"Invalid dataset format: {format_type}. Valid: {valid_formats}"
            )

        # Optional field validation
        if 'checksum' in dataset_info:
            checksum = dataset_info['checksum']
            if not isinstance(checksum, str) or not re.match(r'^[a-f0-9]{32,}$', checksum.lower()):
                raise ValidationError("Invalid checksum format")

        return True

    @classmethod
    def validate_model_architecture(cls, architecture: Dict[str, Any]) -> bool:
        """
        Validate model architecture specification

        Args:
            architecture: Model architecture dictionary

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(architecture, dict):
            raise ValidationError(f"Architecture must be dict, got {type(architecture)}")

        # Required fields for LeZeA models
        required_fields = ['type', 'layers', 'input_size', 'output_size']
        for field in required_fields:
            if field not in architecture:
                raise ValidationError(f"Missing required architecture field: {field}")

        # Validate model type
        model_type = architecture['type']
        valid_types = ['transformer', 'cnn', 'rnn', 'hybrid', 'custom']
        if model_type not in valid_types:
            raise ValidationError(f"Invalid model type: {model_type}. Valid: {valid_types}")

        # Validate layer count
        layers = architecture['layers']
        if not isinstance(layers, int) or layers < 1 or layers > 1000:
            raise ValidationError(f"Invalid layer count: {layers}")

        # Validate input/output sizes
        for size_field in ['input_size', 'output_size']:
            size = architecture[size_field]
            if isinstance(size, int):
                if size < 1 or size > 1_000_000:
                    raise ValidationError(f"Invalid {size_field}: {size}")
            elif isinstance(size, list):
                if not all(isinstance(dim, int) and dim > 0 for dim in size):
                    raise ValidationError(f"Invalid {size_field} dimensions: {size}")
            else:
                raise ValidationError(f"{size_field} must be int or list of ints")

        return True


# Convenience validation functions
def validate_experiment_config(config: Dict[str, Any]) -> bool:
    """Validate complete experiment configuration"""
    validator = LeZeAValidator()

    # Validate experiment metadata
    if 'name' in config:
        validator.validate_experiment_name(config['name'])

    if 'parameters' in config:
        validator.validate_model_params(config['parameters'])

    if 'tags' in config:
        validator.validate_tags(config['tags'])

    if 'architecture' in config:
        validator.validate_model_architecture(config['architecture'])

    return True


def safe_validate(func, *args, **kwargs) -> Tuple[bool, Optional[str]]:
    """
    Safely run validation function and return result

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        func(*args, **kwargs)
        return True, None
    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# --- COMPAT WRAPPERS FOR tracker.py ---
# These wrappers let tracker.py import validate_experiment_name/validate_metrics
# as module-level functions and also RECEIVE the original value back.

def validate_experiment_name(name: str) -> str:
    """Validate and return the original experiment name (raises on failure)."""
    LeZeAValidator.validate_experiment_name(name)
    return name


def validate_metrics(metrics: Dict[str, Union[float, int]]) -> Dict[str, Union[float, int]]:
    """Validate and return the original metrics dict (raises on failure)."""
    LeZeAValidator.validate_metrics(metrics)
    return metrics


# Export key classes and functions
__all__ = [
    'ValidationError',
    'LeZeAValidator',
    'validate_experiment_config',
    'safe_validate',
    # Explicitly export compat wrappers expected by tracker.py
    'validate_experiment_name',
    'validate_metrics',
]
