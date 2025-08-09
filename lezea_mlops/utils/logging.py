"""
Logging utilities for LeZeA MLOps
================================

Centralized logging configuration and utilities:
- Structured logging with JSON formatting
- Multiple output handlers (console, file, remote)
- Performance monitoring and metrics
- Error tracking and alerting
- Experiment-specific log organization
- Log rotation and cleanup

Provides consistent logging across all MLOps components.
"""

import os
import sys
import logging
import logging.handlers
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from collections import defaultdict

# Thread-local storage for context
_context = threading.local()


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, include_fields: List[str] = None):
        super().__init__()
        self.include_fields = include_fields or [
            'timestamp', 'level', 'logger', 'message', 'module', 'function', 'line'
        ]
    
    def format(self, record):
        """Format log record as JSON"""
        try:
            # Base log data
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread': record.thread,
                'thread_name': record.threadName,
                'process': record.process
            }
            
            # Add exception info if present
            if record.exc_info:
                log_data['exception'] = {
                    'type': record.exc_info[0].__name__,
                    'message': str(record.exc_info[1]),
                    'traceback': traceback.format_exception(*record.exc_info)
                }
            
            # Add context information if available
            if hasattr(_context, 'experiment_id'):
                log_data['experiment_id'] = _context.experiment_id
            if hasattr(_context, 'step'):
                log_data['step'] = _context.step
            if hasattr(_context, 'component'):
                log_data['component'] = _context.component
            
            # Add custom fields from record
            for key, value in record.__dict__.items():
                if key.startswith('custom_') and key not in log_data:
                    log_data[key] = value
            
            # Filter fields if specified
            if self.include_fields:
                filtered_data = {k: v for k, v in log_data.items() 
                               if any(field in k for field in self.include_fields)}
                log_data = filtered_data
            
            return json.dumps(log_data, default=str)
            
        except Exception as e:
            # Fallback to simple format if JSON formatting fails
            return f"LOGGING_ERROR: {str(e)} | Original: {record.getMessage()}"


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
    
    def format(self, record):
        """Format log record with colors"""
        try:
            # Create formatted message
            timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            level = record.levelname.ljust(8)
            logger_name = record.name.split('.')[-1]  # Only last part
            message = record.getMessage()
            
            # Add context if available
            context_parts = []
            if hasattr(_context, 'experiment_id'):
                context_parts.append(f"exp:{_context.experiment_id[:8]}")
            if hasattr(_context, 'step'):
                context_parts.append(f"step:{_context.step}")
            if hasattr(_context, 'component'):
                context_parts.append(f"{_context.component}")
            
            context_str = f"[{','.join(context_parts)}]" if context_parts else ""
            
            # Format base message
            formatted = f"{timestamp} {level} {logger_name:15} {context_str:20} {message}"
            
            # Add exception info if present
            if record.exc_info:
                formatted += f"\n{self.formatException(record.exc_info)}"
            
            # Apply colors
            if self.use_colors:
                color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
                reset = self.COLORS['RESET']
                formatted = f"{color}{formatted}{reset}"
            
            return formatted
            
        except Exception as e:
            return f"FORMATTING_ERROR: {str(e)} | Original: {record.getMessage()}"


class ExperimentFileHandler(logging.handlers.RotatingFileHandler):
    """File handler that organizes logs by experiment"""
    
    def __init__(self, base_dir: str, max_bytes: int = 10*1024*1024, backup_count: int = 5):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Start with a default log file
        default_log = self.base_dir / "mlops.log"
        super().__init__(str(default_log), maxBytes=max_bytes, backupCount=backup_count)
        
        self.experiment_handlers = {}
    
    def emit(self, record):
        """Emit log record, potentially to experiment-specific file"""
        try:
            # Check if we have experiment context
            if hasattr(_context, 'experiment_id'):
                experiment_id = _context.experiment_id
                
                # Create experiment-specific handler if needed
                if experiment_id not in self.experiment_handlers:
                    exp_log_path = self.base_dir / f"experiment_{experiment_id[:8]}.log"
                    exp_handler = logging.handlers.RotatingFileHandler(
                        str(exp_log_path), 
                        maxBytes=self.maxBytes,
                        backupCount=self.backupCount
                    )
                    exp_handler.setFormatter(self.formatter)
                    self.experiment_handlers[experiment_id] = exp_handler
                
                # Emit to experiment-specific handler
                self.experiment_handlers[experiment_id].emit(record)
            
            # Always emit to main handler
            super().emit(record)
            
        except Exception as e:
            # Fallback to stderr
            sys.stderr.write(f"Logging error: {e}\n")
            sys.stderr.write(f"Original message: {record.getMessage()}\n")


class PerformanceLogger:
    """Performance monitoring and logging"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
    
    def time_function(self, func_name: str = None):
        """Decorator to time function execution"""
        def decorator(func):
            nonlocal func_name
            if func_name is None:
                func_name = f"{func.__module__}.{func.__name__}"
            
            def wrapper(*args, **kwargs):
                start_time = datetime.now()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Record timing
                    self.timings[func_name].append(duration)
                    self.counters[f"{func_name}_calls"] += 1
                    
                    if success:
                        self.counters[f"{func_name}_success"] += 1
                    else:
                        self.counters[f"{func_name}_errors"] += 1
                    
                    # Log performance info
                    self.logger.debug(
                        f"Function {func_name} took {duration:.3f}s",
                        extra={
                            'custom_performance': True,
                            'custom_function': func_name,
                            'custom_duration': duration,
                            'custom_success': success,
                            'custom_error': error
                        }
                    )
                
                return result
            return wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        summary = {
            'collection_time': datetime.now().isoformat(),
            'function_timings': {},
            'counters': dict(self.counters)
        }
        
        for func_name, times in self.timings.items():
            if times:
                summary['function_timings'][func_name] = {
                    'call_count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'recent_times': times[-10:]  # Last 10 calls
                }
        
        return summary


class AlertManager:
    """Alert management for critical events"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.alert_counts = defaultdict(int)
        self.alert_history = []
        self.alert_thresholds = {
            'ERROR': 10,      # Max 10 errors before alerting
            'CRITICAL': 1,    # Alert immediately on critical
            'performance': 5  # Max 5 performance issues
        }
    
    def check_alert_conditions(self, record: logging.LogRecord):
        """Check if log record should trigger an alert"""
        try:
            # Count by level
            self.alert_counts[record.levelname] += 1
            
            # Check thresholds
            threshold = self.alert_thresholds.get(record.levelname, float('inf'))
            
            if self.alert_counts[record.levelname] >= threshold:
                self._trigger_alert(record.levelname, record)
                # Reset counter after alert
                self.alert_counts[record.levelname] = 0
            
            # Check for specific error patterns
            message = record.getMessage().lower()
            if any(pattern in message for pattern in ['out of memory', 'cuda error', 'connection failed']):
                self._trigger_alert('PATTERN_CRITICAL', record)
            
        except Exception as e:
            self.logger.error(f"Alert checking failed: {e}")
    
    def _trigger_alert(self, alert_type: str, record: logging.LogRecord):
        """Trigger an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'experiment_id': getattr(_context, 'experiment_id', None)
        }
        
        self.alert_history.append(alert)
        
        # Keep only recent alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        # Log the alert
        self.logger.critical(
            f"ALERT TRIGGERED: {alert_type} - {record.getMessage()}",
            extra={'custom_alert': True, 'custom_alert_data': alert}
        )


# Global instances
_performance_logger = None
_alert_manager = None


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    enable_json: bool = False,
    enable_console_colors: bool = True,
    enable_performance: bool = True,
    enable_alerts: bool = True
) -> logging.Logger:
    """
    Setup centralized logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_json: Whether to use JSON formatting for files
        enable_console_colors: Whether to use colored console output
        enable_performance: Whether to enable performance logging
        enable_alerts: Whether to enable alert management
    
    Returns:
        Configured root logger
    """
    global _performance_logger, _alert_manager
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if enable_console_colors:
        console_formatter = ColoredConsoleFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] %(name)-15s %(message)s',
            datefmt='%H:%M:%S'
        )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = ExperimentFileHandler(log_dir)
    if enable_json:
        file_formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] %(name)-20s %(message)s'
        )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    root_logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / "errors.log",
        maxBytes=5*1024*1024,
        backupCount=3
    )
    error_handler.setFormatter(JSONFormatter())
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)
    
    # Setup performance logging
    if enable_performance:
        _performance_logger = PerformanceLogger(root_logger)
    
    # Setup alert management
    if enable_alerts:
        _alert_manager = AlertManager(root_logger)
        
        # Add alert checking to all handlers
        class AlertFilter(logging.Filter):
            def filter(self, record):
                _alert_manager.check_alert_conditions(record)
                return True
        
        alert_filter = AlertFilter()
        for handler in root_logger.handlers:
            handler.addFilter(alert_filter)
    
    # Configure specific loggers
    configure_logger_levels()
    
    root_logger.info("LeZeA MLOps logging initialized")
    root_logger.info(f"Log level: {level}")
    root_logger.info(f"Log directory: {log_path.absolute()}")
    
    return root_logger


def configure_logger_levels():
    """Configure specific logger levels to reduce noise"""
    # Reduce verbosity of external libraries
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('pymongo').setLevel(logging.WARNING)
    logging.getLogger('mlflow').setLevel(logging.WARNING)
    logging.getLogger('git').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name
    
    Args:
        name: Logger name (usually module name)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_experiment_context(experiment_id: str, step: int = None, component: str = None):
    """
    Set experiment context for logging
    
    Args:
        experiment_id: Experiment identifier
        step: Training step (optional)
        component: Component name (optional)
    """
    _context.experiment_id = experiment_id
    if step is not None:
        _context.step = step
    if component is not None:
        _context.component = component


def clear_experiment_context():
    """Clear experiment context"""
    if hasattr(_context, 'experiment_id'):
        delattr(_context, 'experiment_id')
    if hasattr(_context, 'step'):
        delattr(_context, 'step')
    if hasattr(_context, 'component'):
        delattr(_context, 'component')


def log_performance(func_name: str = None):
    """
    Decorator to log function performance
    
    Args:
        func_name: Custom function name for logging
    
    Returns:
        Decorated function
    """
    global _performance_logger
    
    if _performance_logger is None:
        # Create a basic performance logger if not set up
        _performance_logger = PerformanceLogger(get_logger('performance'))
    
    return _performance_logger.time_function(func_name)


def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary from global performance logger"""
    global _performance_logger
    
    if _performance_logger is None:
        return {'error': 'Performance logging not enabled'}
    
    return _performance_logger.get_performance_summary()


def get_alert_history() -> List[Dict[str, Any]]:
    """Get alert history from global alert manager"""
    global _alert_manager
    
    if _alert_manager is None:
        return []
    
    return _alert_manager.alert_history.copy()


def log_structured(
    logger: logging.Logger,
    level: str,
    message: str,
    **kwargs
):
    """
    Log a structured message with additional fields
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **kwargs: Additional fields to include
    """
    # Add custom prefix to extra fields
    extra = {f'custom_{k}': v for k, v in kwargs.items()}
    
    # Get logging method
    log_method = getattr(logger, level.lower())
    log_method(message, extra=extra)


def log_experiment_event(
    logger: logging.Logger,
    event_type: str,
    event_data: Dict[str, Any],
    message: str = None
):
    """
    Log an experiment-specific event
    
    Args:
        logger: Logger instance
        event_type: Type of event (start, end, checkpoint, etc.)
        event_data: Event data dictionary
        message: Optional custom message
    """
    if message is None:
        message = f"Experiment event: {event_type}"
    
    log_structured(
        logger,
        'info',
        message,
        event_type=event_type,
        event_data=event_data,
        experiment_event=True
    )


def log_training_step(
    logger: logging.Logger,
    step: int,
    metrics: Dict[str, Any],
    duration: float = None
):
    """
    Log a training step with metrics
    
    Args:
        logger: Logger instance
        step: Training step number
        metrics: Training metrics dictionary
        duration: Step duration in seconds
    """
    message = f"Training step {step}"
    
    extra_data = {
        'step': step,
        'metrics': metrics,
        'training_step': True
    }
    
    if duration is not None:
        extra_data['duration'] = duration
        message += f" ({duration:.3f}s)"
    
    log_structured(logger, 'info', message, **extra_data)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any] = None,
    message: str = None
):
    """
    Log an error with additional context
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context dictionary
        message: Optional custom message
    """
    if message is None:
        message = f"Error occurred: {type(error).__name__}"
    
    extra_data = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'error_context': context or {}
    }
    
    logger.error(message, exc_info=error, extra={f'custom_{k}': v for k, v in extra_data.items()})


def create_file_logger(
    name: str,
    filepath: str,
    level: str = "INFO",
    format_json: bool = False
) -> logging.Logger:
    """
    Create a dedicated file logger
    
    Args:
        name: Logger name
        filepath: Path to log file
        level: Logging level
        format_json: Whether to use JSON formatting
    
    Returns:
        Configured file logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filepath,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Set formatter
    if format_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
        )
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def cleanup_old_logs(log_dir: str, days_to_keep: int = 30):
    """
    Clean up old log files
    
    Args:
        log_dir: Directory containing log files
        days_to_keep: Number of days of logs to keep
    """
    try:
        log_path = Path(log_dir)
        if not log_path.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        
        for log_file in log_path.glob('*.log*'):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_time:
                    log_file.unlink()
                    deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {log_file}: {e}")
        
        if deleted_count > 0:
            logger = get_logger('log_cleanup')
            logger.info(f"Cleaned up {deleted_count} old log files")
    
    except Exception as e:
        print(f"Log cleanup failed: {e}")


def export_logs(
    log_dir: str,
    output_file: str,
    experiment_id: str = None,
    start_time: datetime = None,
    end_time: datetime = None,
    level_filter: str = None
):
    """
    Export logs to a single file with optional filtering
    
    Args:
        log_dir: Directory containing log files
        output_file: Output file path
        experiment_id: Filter by experiment ID
        start_time: Filter by start time
        end_time: Filter by end time
        level_filter: Filter by log level
    """
    try:
        log_path = Path(log_dir)
        exported_lines = []
        
        # Find relevant log files
        if experiment_id:
            log_files = list(log_path.glob(f"experiment_{experiment_id[:8]}*.log"))
        else:
            log_files = list(log_path.glob("*.log"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Try to parse as JSON for filtering
                        try:
                            log_data = json.loads(line)
                            
                            # Apply filters
                            if level_filter and log_data.get('level') != level_filter.upper():
                                continue
                            
                            if start_time:
                                log_time = datetime.fromisoformat(log_data.get('timestamp', ''))
                                if log_time < start_time:
                                    continue
                            
                            if end_time:
                                log_time = datetime.fromisoformat(log_data.get('timestamp', ''))
                                if log_time > end_time:
                                    continue
                            
                            exported_lines.append(line)
                            
                        except json.JSONDecodeError:
                            # Not JSON, include if no specific filters
                            if not level_filter and not start_time and not end_time:
                                exported_lines.append(line)
                            
            except Exception as e:
                print(f"Failed to read {log_file}: {e}")
        
        # Write exported logs
        with open(output_file, 'w') as f:
            for line in exported_lines:
                f.write(line + '\n')
        
        logger = get_logger('log_export')
        logger.info(f"Exported {len(exported_lines)} log lines to {output_file}")
        
    except Exception as e:
        print(f"Log export failed: {e}")


class LogContext:
    """Context manager for temporary logging context"""
    
    def __init__(self, **context):
        self.context = context
        self.original_context = {}
    
    def __enter__(self):
        # Save original context
        for key in self.context:
            if hasattr(_context, key):
                self.original_context[key] = getattr(_context, key)
        
        # Set new context
        for key, value in self.context.items():
            setattr(_context, key, value)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original context
        for key in self.context:
            if key in self.original_context:
                setattr(_context, key, self.original_context[key])
            elif hasattr(_context, key):
                delattr(_context, key)


# Convenience functions for common logging patterns
def with_experiment_context(experiment_id: str):
    """Context manager for experiment logging"""
    return LogContext(experiment_id=experiment_id)


def with_step_context(step: int):
    """Context manager for step logging"""
    return LogContext(step=step)


def with_component_context(component: str):
    """Context manager for component logging"""
    return LogContext(component=component)


# Module-level convenience logger
logger = get_logger(__name__)