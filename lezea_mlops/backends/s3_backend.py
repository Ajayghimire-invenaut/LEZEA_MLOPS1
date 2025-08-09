"""
S3 Backend for LeZeA MLOps
=========================

Handles storage of large artifacts in AWS S3 including:
- Model checkpoints (tasker and builder models)
- Final trained models with versioning
- Large training artifacts and logs
- Visualization files and reports
- Data exports and backups
- Dataset storage for DVC

This backend provides optimized upload/download with multipart transfers,
intelligent retry logic, and comprehensive metadata management.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
    from botocore.config import Config
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None


class S3Backend:
    """
    AWS S3 backend for large file storage and artifact management
    
    This class provides:
    - Efficient multipart uploads for large files
    - Intelligent retry logic and error handling
    - Metadata management and file integrity checks
    - Lifecycle management and storage optimization
    - Presigned URLs for secure access
    - Batch operations for multiple files
    """
    
    def __init__(self, config):
        """
        Initialize S3 backend
        
        Args:
            config: Configuration object with S3 settings
        """
        if not S3_AVAILABLE:
            raise RuntimeError(
                "AWS SDK is not available. Install with: pip install boto3"
            )
        
        self.config = config
        self.s3_config = config.get_s3_config()
        
        # Validate configuration
        if not self.s3_config.get('aws_access_key_id') or not self.s3_config.get('aws_secret_access_key'):
            raise ValueError("AWS credentials are required. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        
        # Initialize S3 client with optimized configuration
        self._init_s3_client()
        
        # Bucket and prefix configuration
        self.buckets = self.s3_config.get('buckets', {})
        self.prefixes = self.s3_config.get('prefixes', {})
        
        # Verify bucket access
        self._verify_bucket_access()
        
        print(f"✅ S3 backend connected: {self.s3_config['region']}")
    
    def _init_s3_client(self):
        """Initialize S3 client with optimized configuration"""
        try:
            # Configure for optimal performance
            boto_config = Config(
                region_name=self.s3_config['region'],
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'
                },
                max_pool_connections=50,
                s3={
                    'max_bandwidth': None,  # No bandwidth limit
                    'max_concurrency': 10,
                    'multipart_threshold': 64 * 1024 * 1024,  # 64MB
                    'multipart_chunksize': 16 * 1024 * 1024,  # 16MB
                    'use_threads': True
                }
            )
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.s3_config['aws_access_key_id'],
                aws_secret_access_key=self.s3_config['aws_secret_access_key'],
                config=boto_config
            )
            
        except NoCredentialsError:
            raise ValueError("AWS credentials not found or invalid")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize S3 client: {e}")
    
    def _verify_bucket_access(self):
        """Verify access to configured buckets"""
        try:
            # Check artifacts bucket (most important)
            artifacts_bucket = self.buckets.get('artifacts')
            if artifacts_bucket:
                self.s3_client.head_bucket(Bucket=artifacts_bucket)
                print(f"✅ Artifacts bucket accessible: {artifacts_bucket}")
            
            # Check other buckets
            for bucket_type, bucket_name in self.buckets.items():
                if bucket_type != 'artifacts' and bucket_name:
                    try:
                        self.s3_client.head_bucket(Bucket=bucket_name)
                        print(f"✅ {bucket_type} bucket accessible: {bucket_name}")
                    except ClientError as e:
                        print(f"⚠️ {bucket_type} bucket not accessible: {bucket_name} ({e})")
                        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"S3 bucket does not exist: {artifacts_bucket}")
            elif error_code == '403':
                raise ValueError(f"Access denied to S3 bucket: {artifacts_bucket}")
            else:
                raise ConnectionError(f"S3 error: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for integrity verification"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"⚠️ Failed to calculate file hash: {e}")
            return ""
    
    def _get_file_metadata(self, file_path: str, additional_metadata: Dict = None) -> Dict[str, str]:
        """Generate comprehensive metadata for file upload"""
        try:
            file_path = Path(file_path)
            stat = file_path.stat()
            
            metadata = {
                'original_filename': file_path.name,
                'file_size_bytes': str(stat.st_size),
                'file_hash_sha256': self._calculate_file_hash(str(file_path)),
                'upload_timestamp': datetime.now().isoformat(),
                'uploaded_by': 'lezea_mlops',
                'content_type': self._guess_content_type(file_path.suffix)
            }
            
            if additional_metadata:
                # Convert all values to strings for S3 metadata
                for key, value in additional_metadata.items():
                    metadata[f'custom_{key}'] = str(value)
            
            return metadata
            
        except Exception as e:
            print(f"⚠️ Failed to generate file metadata: {e}")
            return {}
    
    def _guess_content_type(self, file_extension: str) -> str:
        """Guess content type based on file extension"""
        content_types = {
            '.pth': 'application/x-pytorch',
            '.pkl': 'application/x-pickle',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.log': 'text/plain',
            '.csv': 'text/csv',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.pdf': 'application/pdf',
            '.html': 'text/html',
            '.zip': 'application/zip',
            '.tar.gz': 'application/gzip'
        }
        return content_types.get(file_extension.lower(), 'application/octet-stream')
    
    def upload_checkpoint(self, local_path: str, experiment_id: str, 
                         step: int = None, metadata: Dict[str, Any] = None) -> str:
        """
        Upload a model checkpoint to S3
        
        Args:
            local_path: Path to local checkpoint file
            experiment_id: Experiment identifier
            step: Training step (optional)
            metadata: Additional metadata
        
        Returns:
            S3 key of uploaded checkpoint
        """
        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Checkpoint file not found: {local_path}")
            
            # Generate S3 key
            filename = os.path.basename(local_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if step is not None:
                s3_key = f"{self.prefixes.get('checkpoints', 'checkpoints/')}{experiment_id}/step_{step}/{timestamp}_{filename}"
            else:
                s3_key = f"{self.prefixes.get('checkpoints', 'checkpoints/')}{experiment_id}/{timestamp}/{filename}"
            
            # Prepare metadata
            upload_metadata = self._get_file_metadata(local_path, {
                'experiment_id': experiment_id,
                'checkpoint_step': step if step is not None else 'unknown',
                'checkpoint_type': 'training_checkpoint',
                **(metadata or {})
            })
            
            # Upload to artifacts bucket
            bucket = self.buckets.get('artifacts')
            if not bucket:
                raise ValueError("Artifacts bucket not configured")
            
            self.s3_client.upload_file(
                local_path,
                bucket,
                s3_key,
                ExtraArgs={
                    'Metadata': upload_metadata,
                    'ContentType': upload_metadata.get('content_type', 'application/octet-stream')
                }
            )
            
            print(f"📤 Uploaded checkpoint: {filename} -> s3://{bucket}/{s3_key}")
            return s3_key
            
        except Exception as e:
            print(f"❌ Failed to upload checkpoint: {e}")
            return None
    
    def upload_final_model(self, local_path: str, experiment_id: str, 
                          model_type: str, model_name: str, 
                          metadata: Dict[str, Any] = None) -> str:
        """
        Upload a final trained model to S3
        
        Args:
            local_path: Path to local model file
            experiment_id: Experiment identifier
            model_type: Type of model (tasker/builder)
            model_name: Name of the model
            metadata: Additional metadata
        
        Returns:
            S3 key of uploaded model
        """
        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Model file not found: {local_path}")
            
            # Generate S3 key for final model
            filename = os.path.basename(local_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{self.prefixes.get('models', 'models/')}{model_type}/{experiment_id}/{model_name}_{timestamp}/{filename}"
            
            # Prepare metadata
            upload_metadata = self._get_file_metadata(local_path, {
                'experiment_id': experiment_id,
                'model_type': model_type,
                'model_name': model_name,
                'model_category': 'final_model',
                **(metadata or {})
            })
            
            # Upload to models bucket or artifacts bucket
            bucket = self.buckets.get('models') or self.buckets.get('artifacts')
            if not bucket:
                raise ValueError("Models bucket not configured")
            
            self.s3_client.upload_file(
                local_path,
                bucket,
                s3_key,
                ExtraArgs={
                    'Metadata': upload_metadata,
                    'ContentType': upload_metadata.get('content_type', 'application/octet-stream')
                }
            )
            
            print(f"🏆 Uploaded final model: {model_name} -> s3://{bucket}/{s3_key}")
            return s3_key
            
        except Exception as e:
            print(f"❌ Failed to upload final model: {e}")
            return None
    
    def upload_artifact(self, local_path: str, artifact_key: str, 
                       bucket_type: str = 'artifacts', metadata: Dict[str, Any] = None) -> str:
        """
        Upload a general artifact to S3
        
        Args:
            local_path: Path to local file
            artifact_key: S3 key for the artifact
            bucket_type: Type of bucket to use
            metadata: Additional metadata
        
        Returns:
            S3 key of uploaded artifact
        """
        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Artifact file not found: {local_path}")
            
            # Get bucket
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")
            
            # Prepare metadata
            upload_metadata = self._get_file_metadata(local_path, {
                'artifact_type': 'general',
                **(metadata or {})
            })
            
            # Upload file
            self.s3_client.upload_file(
                local_path,
                bucket,
                artifact_key,
                ExtraArgs={
                    'Metadata': upload_metadata,
                    'ContentType': upload_metadata.get('content_type', 'application/octet-stream')
                }
            )
            
            print(f"📎 Uploaded artifact: {os.path.basename(local_path)} -> s3://{bucket}/{artifact_key}")
            return artifact_key
            
        except Exception as e:
            print(f"❌ Failed to upload artifact: {e}")
            return None
    
    def download_file(self, s3_key: str, local_path: str, bucket_type: str = 'artifacts', 
                     verify_hash: bool = True) -> bool:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 key of the file
            local_path: Local path to save the file
            bucket_type: Type of bucket to download from
            verify_hash: Whether to verify file integrity
        
        Returns:
            True if download successful
        """
        try:
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(bucket, s3_key, local_path)
            
            # Verify file integrity if requested
            if verify_hash:
                try:
                    # Get stored hash from metadata
                    response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
                    stored_hash = response.get('Metadata', {}).get('file_hash_sha256')
                    
                    if stored_hash:
                        downloaded_hash = self._calculate_file_hash(local_path)
                        if stored_hash != downloaded_hash:
                            print(f"⚠️ File integrity check failed for {s3_key}")
                            return False
                        else:
                            print(f"✅ File integrity verified for {s3_key}")
                            
                except Exception as e:
                    print(f"⚠️ Could not verify file integrity: {e}")
            
            print(f"📥 Downloaded: s3://{bucket}/{s3_key} -> {local_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download file: {e}")
            return False
    
    def list_objects(self, prefix: str = "", bucket_type: str = 'artifacts', 
                    max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket with optional prefix filter
        
        Args:
            prefix: Prefix to filter objects
            bucket_type: Type of bucket to list from
            max_keys: Maximum number of keys to return
        
        Returns:
            List of object information dictionaries
        """
        try:
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            for obj in response.get('Contents', []):
                # Get object metadata
                try:
                    head_response = self.s3_client.head_object(Bucket=bucket, Key=obj['Key'])
                    metadata = head_response.get('Metadata', {})
                except Exception as e:
                    metadata = {}
                
                object_info = {
                    'key': obj['Key'],
                    'size_bytes': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag'].strip('"'),
                    'metadata': metadata,
                    'storage_class': obj.get('StorageClass', 'STANDARD')
                }
                objects.append(object_info)
            
            return sorted(objects, key=lambda x: x['last_modified'], reverse=True)
            
        except Exception as e:
            print(f"❌ Failed to list objects: {e}")
            return []
    
    def list_experiment_artifacts(self, experiment_id: str, artifact_type: str = None) -> List[Dict[str, Any]]:
        """
        List all artifacts for a specific experiment
        
        Args:
            experiment_id: Experiment identifier
            artifact_type: Optional filter by artifact type (checkpoints, models)
        
        Returns:
            List of artifact information
        """
        try:
            artifacts = []
            
            # Search in different prefixes based on artifact type
            if artifact_type is None or artifact_type == 'checkpoints':
                checkpoint_prefix = f"{self.prefixes.get('checkpoints', 'checkpoints/')}{experiment_id}/"
                checkpoints = self.list_objects(checkpoint_prefix, 'artifacts')
                for checkpoint in checkpoints:
                    checkpoint['artifact_type'] = 'checkpoint'
                artifacts.extend(checkpoints)
            
            if artifact_type is None or artifact_type == 'models':
                # Search in both tasker and builder model directories
                for model_type in ['tasker', 'builder']:
                    model_prefix = f"{self.prefixes.get('models', 'models/')}{model_type}/{experiment_id}/"
                    models = self.list_objects(model_prefix, 'models')
                    for model in models:
                        model['artifact_type'] = 'model'
                        model['model_type'] = model_type
                    artifacts.extend(models)
            
            return sorted(artifacts, key=lambda x: x['last_modified'], reverse=True)
            
        except Exception as e:
            print(f"❌ Failed to list experiment artifacts: {e}")
            return []
    
    def delete_object(self, s3_key: str, bucket_type: str = 'artifacts') -> bool:
        """
        Delete an object from S3
        
        Args:
            s3_key: S3 key of the object to delete
            bucket_type: Type of bucket to delete from
        
        Returns:
            True if deletion successful
        """
        try:
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")
            
            self.s3_client.delete_object(Bucket=bucket, Key=s3_key)
            print(f"🗑️ Deleted: s3://{bucket}/{s3_key}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete object: {e}")
            return False
    
    def delete_experiment_artifacts(self, experiment_id: str) -> int:
        """
        Delete all artifacts for an experiment
        
        Args:
            experiment_id: Experiment identifier
        
        Returns:
            Number of objects deleted
        """
        try:
            artifacts = self.list_experiment_artifacts(experiment_id)
            deleted_count = 0
            
            for artifact in artifacts:
                bucket_type = 'models' if artifact.get('artifact_type') == 'model' else 'artifacts'
                if self.delete_object(artifact['key'], bucket_type):
                    deleted_count += 1
            
            print(f"🗑️ Deleted {deleted_count} artifacts for experiment {experiment_id}")
            return deleted_count
            
        except Exception as e:
            print(f"❌ Failed to delete experiment artifacts: {e}")
            return 0
    
    def create_presigned_url(self, s3_key: str, bucket_type: str = 'artifacts', 
                           expiration: int = 3600, method: str = 'get_object') -> str:
        """
        Create a presigned URL for accessing an object
        
        Args:
            s3_key: S3 key of the object
            bucket_type: Type of bucket
            expiration: URL expiration time in seconds
            method: HTTP method (get_object, put_object)
        
        Returns:
            Presigned URL string
        """
        try:
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")
            
            response = self.s3_client.generate_presigned_url(
                method,
                Params={'Bucket': bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            
            print(f"🔗 Generated presigned URL for: {s3_key} (expires in {expiration}s)")
            return response
            
        except Exception as e:
            print(f"❌ Failed to create presigned URL: {e}")
            return None
    
    def get_storage_usage(self, prefix: str = "", bucket_type: str = 'artifacts') -> Dict[str, Any]:
        """
        Get storage usage statistics
        
        Args:
            prefix: Optional prefix to filter objects
            bucket_type: Type of bucket to analyze
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            objects = self.list_objects(prefix, bucket_type, max_keys=10000)
            
            total_size = sum(obj['size_bytes'] for obj in objects)
            object_count = len(objects)
            
            # Group by storage class
            storage_classes = {}
            for obj in objects:
                storage_class = obj.get('storage_class', 'STANDARD')
                if storage_class not in storage_classes:
                    storage_classes[storage_class] = {'count': 0, 'size_bytes': 0}
                storage_classes[storage_class]['count'] += 1
                storage_classes[storage_class]['size_bytes'] += obj['size_bytes']
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2),
                'object_count': object_count,
                'prefix': prefix or 'all',
                'bucket_type': bucket_type,
                'storage_classes': storage_classes
            }
            
        except Exception as e:
            print(f"❌ Failed to get storage usage: {e}")
            return {}
    
    def sync_directory(self, local_dir: str, s3_prefix: str, bucket_type: str = 'artifacts',
                      delete_missing: bool = False, dry_run: bool = False) -> Dict[str, int]:
        """
        Sync a local directory to S3
        
        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix for uploaded files
            bucket_type: Type of bucket to sync to
            delete_missing: Whether to delete S3 objects not in local dir
            dry_run: Whether to only show what would be done
        
        Returns:
            Dictionary with sync statistics
        """
        try:
            local_path = Path(local_dir)
            if not local_path.exists():
                raise FileNotFoundError(f"Local directory does not exist: {local_dir}")
            
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")
            
            stats = {'uploaded': 0, 'skipped': 0, 'deleted': 0, 'errors': 0}
            
            # Upload local files
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    # Calculate relative path and S3 key
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{s3_prefix}{relative_path.as_posix()}"
                    
                    # Check if file needs to be uploaded
                    should_upload = True
                    try:
                        # Check if object exists and compare hash
                        response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
                        stored_hash = response.get('Metadata', {}).get('file_hash_sha256')
                        if stored_hash:
                            local_hash = self._calculate_file_hash(str(file_path))
                            if stored_hash == local_hash:
                                should_upload = False
                                stats['skipped'] += 1
                    except ClientError:
                        # Object doesn't exist, need to upload
                        pass
                    
                    if should_upload:
                        if not dry_run:
                            if self.upload_artifact(str(file_path), s3_key, bucket_type):
                                stats['uploaded'] += 1
                            else:
                                stats['errors'] += 1
                        else:
                            print(f"Would upload: {file_path} -> s3://{bucket}/{s3_key}")
                            stats['uploaded'] += 1
            
            # Delete missing files if requested
            if delete_missing:
                s3_objects = self.list_objects(s3_prefix, bucket_type)
                s3_keys = set(obj['key'] for obj in s3_objects)
                
                # Find local files as S3 keys
                local_keys = set()
                for file_path in local_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        local_keys.add(f"{s3_prefix}{relative_path.as_posix()}")
                
                # Delete missing files
                for s3_key in s3_keys - local_keys:
                    if not dry_run:
                        if self.delete_object(s3_key, bucket_type):
                            stats['deleted'] += 1
                        else:
                            stats['errors'] += 1
                    else:
                        print(f"Would delete: s3://{bucket}/{s3_key}")
                        stats['deleted'] += 1
            
            action = "Would sync" if dry_run else "Synced"
            print(f"📁 {action} directory: {stats}")
            return stats
            
        except Exception as e:
            print(f"❌ Failed to sync directory: {e}")
            return {'uploaded': 0, 'skipped': 0, 'deleted': 0, 'errors': 1}
    
    def get_bucket_info(self, bucket_type: str = 'artifacts') -> Dict[str, Any]:
        """
        Get information about a bucket
        
        Args:
            bucket_type: Type of bucket to get info for
        
        Returns:
            Dictionary with bucket information
        """
        try:
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                return {'error': f"Bucket type '{bucket_type}' not configured"}
            
            # Get bucket location
            try:
                location = self.s3_client.get_bucket_location(Bucket=bucket)
                region = location.get('LocationConstraint') or 'us-east-1'
            except Exception:
                region = self.s3_config['region']
            
            # Get object count and size (limited sample)
            objects = self.list_objects("", bucket_type, max_keys=1000)
            total_size = sum(obj['size_bytes'] for obj in objects)
            
            return {
                'bucket_name': bucket,
                'bucket_type': bucket_type,
                'region': region,
                'sample_object_count': len(objects),
                'sample_total_size_mb': round(total_size / (1024 * 1024), 2),
                'accessible': True
            }
            
        except Exception as e:
            return {
                'bucket_name': bucket,
                'bucket_type': bucket_type,
                'error': str(e),
                'accessible': False
            }
    
    def cleanup_temp_files(self, max_age_days: int = 7) -> int:
        """
        Clean up temporary files older than specified days
        
        Args:
            max_age_days: Maximum age in days for temp files
        
        Returns:
            Number of files deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            temp_prefix = self.prefixes.get('temp', 'temp/')
            temp_objects = self.list_objects(temp_prefix, 'artifacts')
            
            deleted_count = 0
            for obj in temp_objects:
                obj_date = datetime.fromisoformat(obj['last_modified'].replace('Z', '+00:00'))
                if obj_date.replace(tzinfo=None) < cutoff_date:
                    if self.delete_object(obj['key'], 'artifacts'):
                        deleted_count += 1
            
            print(f"🧹 Cleaned up {deleted_count} temp files older than {max_age_days} days")
            return deleted_count
            
        except Exception as e:
            print(f"❌ Failed to cleanup temp files: {e}")
            return 0