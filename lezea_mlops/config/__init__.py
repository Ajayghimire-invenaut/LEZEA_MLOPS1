"""
Configuration management for LeZeA MLOps
Handles all service configurations, paths, and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration manager for all MLOps services
    
    This class:
    - Loads YAML configuration files
    - Manages environment variables
    - Provides service URLs and credentials
    - Ensures directories exist
    """
    
    def __init__(self):
        self.config_dir = Path(__file__).parent
        self.project_root = self.config_dir.parent.parent
        
        # Load YAML configurations
        self.paths = self._load_yaml('paths.yml')
        self.services = self._load_yaml('services.yml')
        
        # Create necessary directories
        self._ensure_directories()
        
        print(f"📁 Configuration loaded from: {self.config_dir}")
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file with error handling"""
        config_path = self.config_dir / filename
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    content = yaml.safe_load(f)
                    return content or {}
            except yaml.YAMLError as e:
                print(f"❌ Error loading {filename}: {e}")
                return {}
        else:
            print(f"⚠️ Config file not found: {filename}")
            return {}
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = self.paths.get('directories', {})
        for dir_name, dir_path in directories.items():
            path = Path(dir_path)
            if not path.is_absolute():
                path = self.project_root / path
            path.mkdir(parents=True, exist_ok=True)
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a specific service"""
        return self.services.get(service_name, {})
    
    def get_mlflow_config(self) -> Dict[str, str]:
        """Get MLflow configuration with environment variable overrides"""
        mlflow_config = self.get_service_config('mlflow')
        
        return {
            'tracking_uri': os.getenv('MLFLOW_TRACKING_URI') or mlflow_config.get('url', 'http://localhost:5000'),
            'backend_store_uri': os.getenv('MLFLOW_BACKEND_STORE_URI') or mlflow_config.get('backend_store_uri'),
            'default_artifact_root': os.getenv('MLFLOW_ARTIFACT_ROOT') or mlflow_config.get('default_artifact_root'),
            'experiment_name': os.getenv('MLFLOW_EXPERIMENT_NAME', 'lezea_experiments')
        }
    
    def get_mongodb_config(self) -> Dict[str, Any]:
        """Get MongoDB configuration with environment variable overrides"""
        mongo_config = self.get_service_config('mongodb')
        
        # Build connection string
        host = os.getenv('MONGO_HOST') or mongo_config.get('host', 'localhost')
        port = os.getenv('MONGO_PORT') or mongo_config.get('port', 27017)
        username = os.getenv('MONGO_USERNAME')
        password = os.getenv('MONGO_PASSWORD')
        
        if username and password:
            connection_string = f"mongodb://{username}:{password}@{host}:{port}/"
        else:
            connection_string = f"mongodb://{host}:{port}/"
        
        return {
            'connection_string': os.getenv('MONGO_CONNECTION_STRING') or connection_string,
            'database': os.getenv('MONGO_DATABASE') or mongo_config.get('database', 'lezea_mlops'),
            'collections': mongo_config.get('collections', {})
        }
    
    def get_s3_config(self) -> Dict[str, Any]:
        """Get S3 configuration with environment variable overrides"""
        s3_config = self.get_service_config('s3')
        
        return {
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'region': os.getenv('AWS_REGION') or s3_config.get('region', 'us-east-1'),
            'buckets': s3_config.get('buckets', {}),
            'prefixes': s3_config.get('prefixes', {})
        }
    
    def get_postgres_config(self) -> Dict[str, Any]:
        """Get PostgreSQL configuration with environment variable overrides"""
        postgres_config = self.get_service_config('postgres')
        
        return {
            'host': os.getenv('POSTGRES_HOST') or postgres_config.get('host', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT') or postgres_config.get('port', 5432)),
            'database': os.getenv('POSTGRES_DATABASE') or postgres_config.get('database', 'mlflow_db'),
            'user': os.getenv('POSTGRES_USER') or postgres_config.get('user', 'mlflow_user'),
            'password': os.getenv('POSTGRES_PASSWORD') or postgres_config.get('password', 'mlflow_password123')
        }
    
    def get_path(self, path_name: str) -> Path:
        """Get path for a specific directory or file pattern"""
        directories = self.paths.get('directories', {})
        if path_name in directories:
            path = Path(directories[path_name])
            if not path.is_absolute():
                path = self.project_root / path
            return path
        return self.project_root / path_name
    
    def get_file_pattern(self, pattern_name: str, **kwargs) -> str:
        """Get file pattern with variable substitution"""
        patterns = self.paths.get('patterns', {})
        if pattern_name in patterns:
            return patterns[pattern_name].format(**kwargs)
        return pattern_name
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate that all required configurations are present"""
        validation_results = {}
        
        # Check MLflow config
        mlflow_config = self.get_mlflow_config()
        validation_results['mlflow'] = bool(mlflow_config.get('tracking_uri'))
        
        # Check MongoDB config
        mongo_config = self.get_mongodb_config()
        validation_results['mongodb'] = bool(mongo_config.get('connection_string'))
        
        # Check S3 config
        s3_config = self.get_s3_config()
        validation_results['s3'] = bool(
            s3_config.get('aws_access_key_id') and 
            s3_config.get('aws_secret_access_key')
        )
        
        # Check PostgreSQL config
        postgres_config = self.get_postgres_config()
        validation_results['postgres'] = bool(
            postgres_config.get('host') and 
            postgres_config.get('database')
        )
        
        return validation_results
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print("\n🔧 LeZeA MLOps Configuration Summary")
        print("=" * 50)
        
        # MLflow
        mlflow_config = self.get_mlflow_config()
        print(f"MLflow: {mlflow_config.get('tracking_uri')}")
        
        # MongoDB
        mongo_config = self.get_mongodb_config()
        print(f"MongoDB: {mongo_config.get('database')} @ {mongo_config.get('connection_string').split('@')[-1] if '@' in mongo_config.get('connection_string', '') else mongo_config.get('connection_string')}")
        
        # S3
        s3_config = self.get_s3_config()
        print(f"S3: {s3_config.get('region')} region")
        
        # PostgreSQL
        postgres_config = self.get_postgres_config()
        print(f"PostgreSQL: {postgres_config.get('database')} @ {postgres_config.get('host')}:{postgres_config.get('port')}")
        
        # Validation
        validation = self.validate_config()
        print(f"\nValidation: {sum(validation.values())}/{len(validation)} services configured")
        for service, is_valid in validation.items():
            status = "✅" if is_valid else "❌"
            print(f"  {status} {service}")
        
        print("=" * 50)

# Global configuration instance
config = Config()

# Export for easy importing
__all__ = ['Config', 'config']