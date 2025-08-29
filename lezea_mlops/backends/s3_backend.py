"""
S3 Backend for LeZeA MLOps System

This module provides a comprehensive AWS S3 backend for artifact storage, model management,
and data archiving in machine learning workflows. It supports S3-compatible storage systems
including MinIO and other cloud providers with full feature compatibility.

Features:
    - Multi-bucket architecture for organized artifact storage
    - Multipart upload with progress tracking for large files
    - Server-side encryption (SSE-S3, SSE-KMS) support
    - Integrity verification using SHA256 checksums
    - Presigned URL generation for secure access
    - Directory synchronization with diff-based uploads
    - Storage usage analytics and cost optimization
    - Comprehensive error handling and retry logic
    - S3 Transfer Acceleration and endpoint flexibility

Author: [Your Name/Team]
Date: 2025-08-29
Version: 1.0.0
License: [Your License]
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# AWS SDK dependencies with graceful fallback
try:
    import boto3
    from botocore.config import Config as BotoCoreConfig
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    from boto3.s3.transfer import TransferConfig as S3TransferConfig
    S3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore
    BotoCoreConfig = object  # type: ignore
    S3TransferConfig = object  # type: ignore
    ClientError = NoCredentialsError = BotoCoreError = Exception  # type: ignore
    S3_AVAILABLE = False


def _determine_content_type(file_path: str) -> str:
    """
    Determine MIME content type for a file path.
    
    Uses Python's mimetypes module for accurate type detection
    including compressed formats like .tar.gz.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string, defaults to 'application/octet-stream'
    """
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type or "application/octet-stream"


def _calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file for integrity verification.
    
    Args:
        file_path: Path object pointing to the file
        
    Returns:
        SHA256 hash as hexadecimal string, empty string on error
    """
    hasher = hashlib.sha256()
    try:
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


class S3Backend:
    """
    Professional AWS S3 backend for artifact and model storage.
    
    This class provides a comprehensive interface for S3 operations with
    enterprise-grade features including encryption, integrity checks,
    multipart uploads, and advanced storage management.
    
    Key Features:
        - Multi-bucket architecture for organized storage
        - Production-ready multipart upload with progress tracking
        - Server-side encryption with KMS support
        - Integrity verification using SHA256 checksums
        - S3-compatible endpoint support (MinIO, etc.)
        - Presigned URL generation for secure access
        - Directory synchronization with differential updates
        - Storage analytics and usage monitoring
        
    Attributes:
        config: Configuration object containing S3 settings
        s3_config: Parsed S3 configuration dictionary
        s3_client: Boto3 S3 client instance
        available: Boolean indicating backend availability
        buckets: Dictionary mapping bucket types to names
        prefixes: Dictionary mapping content types to S3 prefixes
    """

    def __init__(self, config):
        """
        Initialize S3 backend with configuration and client setup.
        
        Args:
            config: Configuration object with get_s3_config() method
            
        Raises:
            RuntimeError: If boto3 is not available
            ValueError: If AWS credentials are missing
            ConnectionError: If S3 client initialization fails
        """
        if not S3_AVAILABLE:
            raise RuntimeError("boto3 not available. Install with: pip install boto3")

        self.config = config
        self.s3_config = config.get_s3_config()

        # Extract configuration settings
        self.region: str = self.s3_config.get("region", "us-east-1")
        self.endpoint_url: Optional[str] = self.s3_config.get("endpoint_url")
        self.use_accelerate: bool = bool(self.s3_config.get("accelerate", False))
        self.server_side_encryption: Optional[str] = self.s3_config.get("sse")
        self.kms_key_id: Optional[str] = self.s3_config.get("kms_key_id")
        self.storage_class: Optional[str] = self.s3_config.get("storage_class")

        # Storage organization
        self.buckets: Dict[str, str] = self.s3_config.get("buckets", {})
        self.prefixes: Dict[str, str] = self.s3_config.get("prefixes", {})

        # Initialize S3 client
        self._initialize_s3_client()
        self._verify_bucket_access()

        self.available = True
        endpoint_display = self.endpoint_url or 'aws'
        print(f"S3 backend connected (region={self.region}, endpoint={endpoint_display})")

    def _initialize_s3_client(self):
        """Initialize boto3 S3 client with production-ready configuration."""
        # Prepare session credentials (use default credential chain when possible)
        session_kwargs = {}
        if (self.s3_config.get("aws_access_key_id") and 
            self.s3_config.get("aws_secret_access_key")):
            session_kwargs.update(
                aws_access_key_id=self.s3_config["aws_access_key_id"],
                aws_secret_access_key=self.s3_config["aws_secret_access_key"],
                aws_session_token=self.s3_config.get("aws_session_token"),
            )

        # Configure boto3 client settings
        self._boto_config = BotoCoreConfig(
            region_name=self.region,
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=int(self.s3_config.get("max_pool_connections", 50)),
            signature_version=self.s3_config.get("signature_version", "s3v4"),
        )

        # Configure transfer settings for multipart uploads
        self._transfer_config = S3TransferConfig(
            multipart_threshold=int(self.s3_config.get("multipart_threshold", 64 * 1024 * 1024)),
            multipart_chunksize=int(self.s3_config.get("multipart_chunksize", 16 * 1024 * 1024)),
            max_concurrency=int(self.s3_config.get("max_concurrency", 10)),
            use_threads=True,
        )

        try:
            # Create S3 client
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                config=self._boto_config,
                **session_kwargs,
            )

            # Configure Transfer Acceleration if requested
            if self.use_accelerate:
                self._enable_transfer_acceleration()

        except NoCredentialsError as e:
            raise ValueError("No AWS credentials found (environment/role/credentials file)") from e
        except Exception as e:
            raise ConnectionError(f"Failed to initialize S3 client: {e}") from e

    def _enable_transfer_acceleration(self):
        """Enable S3 Transfer Acceleration for configured buckets."""
        try:
            for bucket_name in set(self.buckets.values()):
                if bucket_name:
                    self.s3_client.put_bucket_accelerate_configuration(
                        Bucket=bucket_name,
                        AccelerateConfiguration={"Status": "Enabled"}
                    )
        except Exception:
            # Transfer acceleration may not be supported by all providers
            pass

    # Health Monitoring and Connectivity

    def ping(self) -> bool:
        """
        Test S3 backend connectivity and health.
        
        Used by health monitoring systems to verify backend availability.
        
        Returns:
            True if backend is accessible and operational, False otherwise
        """
        try:
            # Test primary artifacts bucket if configured
            artifacts_bucket = self.buckets.get("artifacts")
            if artifacts_bucket:
                self.s3_client.head_bucket(Bucket=artifacts_bucket)
                return True

            # Fallback to general bucket listing
            self.s3_client.list_buckets()
            return True

        except Exception:
            return False

    def _verify_bucket_access(self) -> None:
        """
        Verify access to configured S3 buckets during initialization.
        
        Raises:
            ConnectionError: If primary bucket access verification fails
        """
        try:
            # Verify primary artifacts bucket
            artifacts_bucket = self.buckets.get("artifacts")
            if artifacts_bucket:
                self.s3_client.head_bucket(Bucket=artifacts_bucket)
                print(f"Artifacts bucket verified: {artifacts_bucket}")

            # Test additional buckets with non-fatal warnings
            for bucket_type, bucket_name in self.buckets.items():
                if not bucket_name or bucket_type == "artifacts":
                    continue

                try:
                    self.s3_client.head_bucket(Bucket=bucket_name)
                    print(f"{bucket_type} bucket verified: {bucket_name}")
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code")
                    print(f"{bucket_type} bucket not accessible ({bucket_name}): {error_code}")

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            raise ConnectionError(f"S3 bucket access error: {error_code}") from e

    # Metadata and Upload Utilities

    def _create_file_metadata(self, file_path: Path, custom_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Create comprehensive metadata for S3 object storage.
        
        Args:
            file_path: Path to the file being uploaded
            custom_metadata: Optional custom metadata to include
            
        Returns:
            Dictionary of metadata key-value pairs
        """
        metadata: Dict[str, str] = {
            "original_filename": file_path.name,
            "file_size_bytes": str(file_path.stat().st_size if file_path.exists() else 0),
            "file_hash_sha256": _calculate_file_hash(file_path) if file_path.exists() else "",
            "upload_timestamp": datetime.now(timezone.utc).isoformat(),
            "uploaded_by": "lezea_mlops",
        }

        # Add custom metadata with prefix to avoid conflicts
        if custom_metadata:
            for key, value in custom_metadata.items():
                metadata[f"custom_{key}"] = str(value)

        return metadata

    def _create_upload_arguments(self, content_type: str, metadata: Dict[str, str]) -> Dict[str, Any]:
        """
        Create S3 upload arguments including encryption and storage settings.
        
        Args:
            content_type: MIME type for the content
            metadata: Metadata dictionary for the object
            
        Returns:
            Dictionary of S3 upload arguments
        """
        upload_args: Dict[str, Any] = {
            "ContentType": content_type,
            "Metadata": metadata
        }

        # Add storage class if specified
        if self.storage_class:
            upload_args["StorageClass"] = self.storage_class

        # Configure server-side encryption
        if self.server_side_encryption:
            upload_args["ServerSideEncryption"] = self.server_side_encryption
            if (self.server_side_encryption == "aws:kms" and self.kms_key_id):
                upload_args["SSEKMSKeyId"] = self.kms_key_id

        return upload_args

    # Core Upload Operations

    def upload_checkpoint(
        self,
        local_path: str,
        experiment_id: str,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Optional[str]:
        """
        Upload training checkpoint with organized storage structure.
        
        Args:
            local_path: Path to checkpoint file
            experiment_id: Unique experiment identifier
            step: Optional training step number
            metadata: Optional additional metadata
            progress_callback: Optional progress tracking function
            
        Returns:
            S3 key of uploaded checkpoint or None if failed
        """
        try:
            checkpoint_path = Path(local_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {local_path}")

            # Generate organized S3 key
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = checkpoint_path.name
            prefix = self.prefixes.get("checkpoints", "checkpoints/")

            if step is not None:
                s3_key = f"{prefix}{experiment_id}/step_{int(step)}/{timestamp}_{filename}"
            else:
                s3_key = f"{prefix}{experiment_id}/{timestamp}/{filename}"

            # Prepare metadata
            file_metadata = self._create_file_metadata(
                checkpoint_path,
                {
                    "experiment_id": experiment_id,
                    "checkpoint_step": step or -1,
                    "checkpoint_type": "training_checkpoint",
                    **(metadata or {})
                }
            )

            # Upload to artifacts bucket
            artifacts_bucket = self.buckets.get("artifacts")
            if not artifacts_bucket:
                raise ValueError("Artifacts bucket not configured")

            self._execute_file_upload(
                artifacts_bucket, s3_key, checkpoint_path, file_metadata, progress_callback
            )

            print(f"Uploaded checkpoint: {filename} -> s3://{artifacts_bucket}/{s3_key}")
            return s3_key

        except Exception as e:
            print(f"Failed to upload checkpoint: {e}")
            return None

    def upload_final_model(
        self,
        local_path: str,
        experiment_id: str,
        model_type: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Optional[str]:
        """
        Upload final trained model with comprehensive metadata.
        
        Args:
            local_path: Path to model file
            experiment_id: Unique experiment identifier
            model_type: Type/category of model
            model_name: Human-readable model name
            metadata: Optional additional metadata
            progress_callback: Optional progress tracking function
            
        Returns:
            S3 key of uploaded model or None if failed
        """
        try:
            model_path = Path(local_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {local_path}")

            # Generate organized S3 key
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = model_path.name
            prefix = self.prefixes.get("models", "models/")
            s3_key = f"{prefix}{model_type}/{experiment_id}/{model_name}_{timestamp}/{filename}"

            # Prepare comprehensive model metadata
            file_metadata = self._create_file_metadata(
                model_path,
                {
                    "experiment_id": experiment_id,
                    "model_type": model_type,
                    "model_name": model_name,
                    "model_category": "final_model",
                    **(metadata or {})
                }
            )

            # Use models bucket if available, fallback to artifacts
            target_bucket = self.buckets.get("models") or self.buckets.get("artifacts")
            if not target_bucket:
                raise ValueError("Models or artifacts bucket not configured")

            self._execute_file_upload(
                target_bucket, s3_key, model_path, file_metadata, progress_callback
            )

            print(f"Uploaded final model: {model_name} -> s3://{target_bucket}/{s3_key}")
            return s3_key

        except Exception as e:
            print(f"Failed to upload final model: {e}")
            return None

    def _execute_file_upload(
        self,
        bucket: str,
        key: str,
        file_path: Path,
        metadata: Dict[str, str],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """
        Execute file upload with transfer configuration and progress tracking.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            file_path: Local file path
            metadata: Object metadata
            progress_callback: Optional progress callback function
        """
        class ProgressTracker:
            """Progress tracking wrapper for upload callbacks."""
            def __init__(self, callback: Optional[Callable[[int], None]]):
                self.callback = callback
                self.bytes_transferred = 0

            def __call__(self, bytes_amount: int):
                if self.callback:
                    self.bytes_transferred += bytes_amount
                    try:
                        self.callback(self.bytes_transferred)
                    except Exception:
                        # Don't let callback errors interrupt upload
                        pass

        # Prepare upload arguments
        upload_args = self._create_upload_arguments(
            _determine_content_type(str(file_path)), metadata
        )

        # Execute upload with transfer configuration
        self.s3_client.upload_file(
            Filename=str(file_path),
            Bucket=bucket,
            Key=key,
            ExtraArgs=upload_args,
            Callback=ProgressTracker(progress_callback),
            Config=self._transfer_config,
        )

    # Generic Artifact Operations

    def upload_artifact(
        self,
        local_path: str,
        artifact_key: str,
        bucket_type: str = "artifacts",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Upload generic artifact to specified bucket type.
        
        Args:
            local_path: Path to local artifact file
            artifact_key: S3 key for the artifact
            bucket_type: Type of bucket to upload to
            metadata: Optional additional metadata
            
        Returns:
            S3 key if successful, None if failed
        """
        try:
            artifact_path = Path(local_path)
            if not artifact_path.exists():
                raise FileNotFoundError(f"Artifact not found: {local_path}")

            target_bucket = self.buckets.get(bucket_type)
            if not target_bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            # Prepare metadata
            file_metadata = self._create_file_metadata(
                artifact_path,
                {"artifact_type": "general", **(metadata or {})}
            )

            self._execute_file_upload(target_bucket, artifact_key, artifact_path, file_metadata, None)
            print(f"Uploaded artifact: {artifact_path.name} -> s3://{target_bucket}/{artifact_key}")
            return artifact_key

        except Exception as e:
            print(f"Failed to upload artifact: {e}")
            return None

    # Directory Synchronization

    def sync_directory(
        self,
        local_directory: str,
        s3_prefix: str,
        bucket_type: str = "artifacts",
        delete_missing: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        Synchronize local directory with S3 prefix using differential updates.
        
        Compares local files with remote objects using SHA256 hashes to
        determine which files need uploading, updating, or deletion.
        
        Args:
            local_directory: Path to local directory to sync
            s3_prefix: S3 prefix to sync with
            bucket_type: Type of bucket for synchronization
            delete_missing: Whether to delete remote objects not in local directory
            dry_run: Whether to simulate sync without making changes
            
        Returns:
            Dictionary with sync operation statistics
        """
        sync_stats = {"uploaded": 0, "skipped": 0, "deleted": 0, "errors": 0}

        try:
            local_root = Path(local_directory)
            if not local_root.exists():
                raise FileNotFoundError(f"Local directory not found: {local_directory}")

            target_bucket = self.buckets.get(bucket_type)
            if not target_bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            # Upload/update phase - compare local files with remote
            for local_file in local_root.rglob("*"):
                if not local_file.is_file():
                    continue

                # Generate S3 key for local file
                relative_path = local_file.relative_to(local_root).as_posix()
                s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}"

                try:
                    # Check if file exists remotely and compare hashes
                    should_upload = self._should_upload_file(target_bucket, s3_key, local_file)

                    if not should_upload:
                        sync_stats["skipped"] += 1
                        continue

                    if dry_run:
                        print(f"[DRY RUN] Upload {local_file} -> s3://{target_bucket}/{s3_key}")
                        sync_stats["uploaded"] += 1
                    else:
                        if self.upload_artifact(str(local_file), s3_key, bucket_type):
                            sync_stats["uploaded"] += 1
                        else:
                            sync_stats["errors"] += 1

                except Exception:
                    sync_stats["errors"] += 1

            # Deletion phase - remove remote objects not in local directory
            if delete_missing:
                self._sync_delete_missing(
                    target_bucket, s3_prefix, local_root, bucket_type, dry_run, sync_stats
                )

            action_word = "Would sync" if dry_run else "Synced"
            print(f"Directory synchronization completed: {action_word} {sync_stats}")
            return sync_stats

        except Exception as e:
            print(f"Failed to sync directory: {e}")
            sync_stats["errors"] += 1
            return sync_stats

    def _should_upload_file(self, bucket: str, s3_key: str, local_file: Path) -> bool:
        """
        Determine if local file should be uploaded based on hash comparison.
        
        Args:
            bucket: S3 bucket name
            s3_key: S3 object key
            local_file: Local file path
            
        Returns:
            True if file should be uploaded, False if unchanged
        """
        try:
            # Get remote object metadata
            head_response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            stored_hash = head_response.get("Metadata", {}).get("file_hash_sha256")

            if stored_hash:
                # Compare with local file hash
                local_hash = _calculate_file_hash(local_file)
                return stored_hash != local_hash

            # Upload if no stored hash found
            return True

        except ClientError:
            # Object doesn't exist, should upload
            return True

    def _sync_delete_missing(
        self,
        bucket: str,
        s3_prefix: str,
        local_root: Path,
        bucket_type: str,
        dry_run: bool,
        sync_stats: Dict[str, int],
    ) -> None:
        """
        Delete remote objects that don't exist in local directory.
        
        Args:
            bucket: S3 bucket name
            s3_prefix: S3 prefix being synced
            local_root: Local directory root
            bucket_type: Bucket type for operations
            dry_run: Whether this is a dry run
            sync_stats: Statistics dictionary to update
        """
        try:
            # Get all remote objects
            remote_objects = self.list_objects(s3_prefix, bucket_type, max_keys=10000)
            remote_keys = {obj["key"] for obj in remote_objects}

            # Get all local file keys
            local_keys = set()
            for local_file in local_root.rglob("*"):
                if local_file.is_file():
                    relative_path = local_file.relative_to(local_root).as_posix()
                    local_keys.add(f"{s3_prefix.rstrip('/')}/{relative_path}")

            # Delete objects that don't exist locally
            objects_to_delete = remote_keys - local_keys
            for s3_key in objects_to_delete:
                if dry_run:
                    print(f"[DRY RUN] Delete s3://{bucket}/{s3_key}")
                    sync_stats["deleted"] += 1
                else:
                    if self.delete_object(s3_key, bucket_type):
                        sync_stats["deleted"] += 1
                    else:
                        sync_stats["errors"] += 1

        except Exception:
            sync_stats["errors"] += 1

    # Bucket Information and Maintenance

    def get_bucket_info(self, bucket_type: str = "artifacts") -> Dict[str, Any]:
        """
        Get comprehensive information about a configured bucket.
        
        Args:
            bucket_type: Type of bucket to analyze
            
        Returns:
            Dictionary containing bucket information and statistics
        """
        try:
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                return {
                    "error": f"Bucket type '{bucket_type}' not configured",
                    "accessible": False
                }

            # Get bucket location
            try:
                location_response = self.s3_client.get_bucket_location(Bucket=bucket_name)
                bucket_region = location_response.get("LocationConstraint") or "us-east-1"
            except Exception:
                bucket_region = self.region

            # Get sample statistics
            sample_objects = self.list_objects("", bucket_type, max_keys=500)
            total_sample_size = sum(obj["size_bytes"] for obj in sample_objects)

            return {
                "bucket_name": bucket_name,
                "bucket_type": bucket_type,
                "region": bucket_region,
                "sample_object_count": len(sample_objects),
                "sample_total_size_mb": round(total_sample_size / (1024 * 1024), 2),
                "accessible": True,
            }

        except Exception as e:
            return {
                "bucket_type": bucket_type,
                "error": str(e),
                "accessible": False
            }

    def cleanup_temporary_files(self, max_age_days: int = 7) -> int:
        """
        Clean up temporary files older than specified age.
        
        Args:
            max_age_days: Maximum age in days for temporary files
            
        Returns:
            Number of files successfully cleaned up
        """
        try:
            # Calculate cutoff timestamp
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            temp_prefix = self.prefixes.get("temp", "temp/")
            
            # List temporary objects
            temp_objects = self.list_objects(temp_prefix, "artifacts", max_keys=10000)
            cleanup_count = 0

            for temp_object in temp_objects:
                try:
                    # Parse object modification time
                    object_time_str = temp_object["last_modified"]
                    object_time = datetime.fromisoformat(object_time_str.replace("Z", "+00:00"))

                    # Delete if older than cutoff
                    if object_time < cutoff_time:
                        if self.delete_object(temp_object["key"], "artifacts"):
                            cleanup_count += 1

                except Exception:
                    # Skip objects with unparseable timestamps
                    continue

            print(f"Cleaned up {cleanup_count} temporary objects older than {max_age_days} days")
            return cleanup_count

        except Exception as e:
            print(f"Failed to cleanup temporary files: {e}")
            return 0

    # Advanced Operations

    def create_bucket_lifecycle_policy(self, bucket_type: str = "artifacts") -> bool:
        """
        Create intelligent lifecycle policy for cost optimization.
        
        Args:
            bucket_type: Type of bucket to configure
            
        Returns:
            True if policy created successfully, False otherwise
        """
        try:
            target_bucket = self.buckets.get(bucket_type)
            if not target_bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            # Define lifecycle rules for cost optimization
            lifecycle_rules = [
                {
                    "ID": "ArchiveOldCheckpoints",
                    "Status": "Enabled",
                    "Filter": {"Prefix": self.prefixes.get("checkpoints", "checkpoints/")},
                    "Transitions": [
                        {
                            "Days": 30,
                            "StorageClass": "STANDARD_IA"
                        },
                        {
                            "Days": 90,
                            "StorageClass": "GLACIER"
                        }
                    ]
                },
                {
                    "ID": "DeleteTempFiles",
                    "Status": "Enabled",
                    "Filter": {"Prefix": self.prefixes.get("temp", "temp/")},
                    "Expiration": {"Days": 7}
                }
            ]

            # Apply lifecycle configuration
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=target_bucket,
                LifecycleConfiguration={"Rules": lifecycle_rules}
            )

            print(f"Lifecycle policy created for {target_bucket}")
            return True

        except Exception as e:
            print(f"Failed to create lifecycle policy: {e}")
            return False

    def batch_upload_files(
        self,
        file_paths: List[str],
        s3_prefix: str,
        bucket_type: str = "artifacts",
        max_workers: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Upload multiple files concurrently with progress tracking.
        
        Args:
            file_paths: List of local file paths to upload
            s3_prefix: S3 prefix for uploaded files
            bucket_type: Type of bucket to upload to
            max_workers: Maximum concurrent upload threads
            progress_callback: Optional callback for progress updates (completed, total)
            
        Returns:
            Dictionary with upload results and statistics
        """
        import concurrent.futures
        import threading

        upload_results = {
            "successful": [],
            "failed": [],
            "total_files": len(file_paths),
            "total_size_mb": 0.0
        }

        completed_count = threading.local()
        completed_count.value = 0
        lock = threading.Lock()

        def upload_single_file(file_path: str) -> Tuple[str, bool, Optional[str]]:
            """Upload single file and return result."""
            try:
                file_obj = Path(file_path)
                if not file_obj.exists():
                    return file_path, False, "File not found"

                # Generate S3 key
                filename = file_obj.name
                s3_key = f"{s3_prefix.rstrip('/')}/{filename}"

                # Upload file
                result = self.upload_artifact(file_path, s3_key, bucket_type)
                
                # Update progress
                with lock:
                    completed_count.value += 1
                    if progress_callback:
                        try:
                            progress_callback(completed_count.value, len(file_paths))
                        except Exception:
                            pass

                return file_path, result is not None, result

            except Exception as e:
                return file_path, False, str(e)

        try:
            # Calculate total size
            total_size = 0
            for file_path in file_paths:
                try:
                    total_size += Path(file_path).stat().st_size
                except Exception:
                    pass
            upload_results["total_size_mb"] = round(total_size / (1024 * 1024), 2)

            # Execute concurrent uploads
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(upload_single_file, file_path): file_path
                    for file_path in file_paths
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    file_path, success, result = future.result()
                    
                    if success:
                        upload_results["successful"].append({
                            "file_path": file_path,
                            "s3_key": result
                        })
                    else:
                        upload_results["failed"].append({
                            "file_path": file_path,
                            "error": result
                        })

            success_count = len(upload_results["successful"])
            failure_count = len(upload_results["failed"])
            
            print(f"Batch upload completed: {success_count} successful, {failure_count} failed")
            return upload_results

        except Exception as e:
            print(f"Failed to execute batch upload: {e}")
            upload_results["failed"] = [{"file_path": fp, "error": str(e)} for fp in file_paths]
            return upload_results

    def generate_download_manifest(
        self, 
        experiment_id: str, 
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Generate manifest file for experiment artifacts download.
        
        Args:
            experiment_id: Unique experiment identifier
            output_file: Optional file path to save manifest
            
        Returns:
            Dictionary containing download manifest data
        """
        try:
            # Get all experiment artifacts
            artifacts = self.list_experiment_artifacts(experiment_id)
            
            # Create manifest structure
            manifest = {
                "experiment_id": experiment_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_artifacts": len(artifacts),
                "total_size_bytes": sum(artifact["size_bytes"] for artifact in artifacts),
                "artifacts": []
            }

            # Process each artifact
            for artifact in artifacts:
                # Generate presigned URL for download
                presigned_url = self.create_presigned_url(
                    artifact["key"],
                    bucket_type="models" if artifact.get("artifact_type") == "model" and self.buckets.get("models") else "artifacts",
                    expiration_seconds=3600 * 24  # 24 hours
                )

                manifest["artifacts"].append({
                    "key": artifact["key"],
                    "artifact_type": artifact.get("artifact_type", "unknown"),
                    "size_bytes": artifact["size_bytes"],
                    "last_modified": artifact["last_modified"],
                    "download_url": presigned_url,
                    "metadata": artifact.get("metadata", {})
                })

            # Calculate summary statistics
            manifest["total_size_mb"] = round(manifest["total_size_bytes"] / (1024 * 1024), 2)
            manifest["total_size_gb"] = round(manifest["total_size_bytes"] / (1024 * 1024 * 1024), 2)

            # Save manifest file if requested
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, indent=2)
                print(f"Download manifest saved: {output_file}")

            return manifest

        except Exception as e:
            print(f"Failed to generate download manifest: {e}")
            return {"error": str(e)}

    # Resource Management and Cleanup

    def close(self) -> None:
        """
        Clean up S3 backend resources and connections.
        
        Properly closes any open connections and releases resources
        for clean application shutdown.
        """
        # Boto3 clients are generally stateless and don't require explicit cleanup
        print("S3 backend cleaned up")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic resource cleanup."""
        self.close()

    # Download and Retrieval Operations

    def download_file(
        self,
        s3_key: str,
        local_path: str,
        bucket_type: str = "artifacts",
        verify_integrity: bool = True,
    ) -> bool:
        """
        Download file from S3 with optional integrity verification.
        
        Args:
            s3_key: S3 object key
            local_path: Local destination path
            bucket_type: Type of bucket to download from
            verify_integrity: Whether to verify file integrity using stored hash
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            target_bucket = self.buckets.get(bucket_type)
            if not target_bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            # Create local directory if needed
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            # Download file
            self.s3_client.download_file(target_bucket, s3_key, local_path)

            # Verify integrity if requested
            if verify_integrity:
                self._verify_download_integrity(target_bucket, s3_key, local_path)

            print(f"Downloaded: s3://{target_bucket}/{s3_key} -> {local_path}")
            return True

        except Exception as e:
            print(f"Failed to download file: {e}")
            return False

    def _verify_download_integrity(self, bucket: str, s3_key: str, local_path: str) -> None:
        """
        Verify downloaded file integrity using stored SHA256 hash.
        
        Args:
            bucket: S3 bucket name
            s3_key: S3 object key
            local_path: Local file path to verify
        """
        try:
            # Retrieve stored hash from object metadata
            head_response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            stored_hash = head_response.get("Metadata", {}).get("file_hash_sha256")

            if stored_hash:
                # Calculate local file hash
                local_hash = _calculate_file_hash(Path(local_path))
                
                if stored_hash != local_hash:
                    print(f"Integrity verification failed for {s3_key}")
                else:
                    print(f"Integrity verification passed for {s3_key}")

        except Exception:
            # Integrity check is best-effort, don't fail download
            pass

    # Object Listing and Discovery

    def list_objects(
        self,
        prefix: str = "",
        bucket_type: str = "artifacts",
        max_keys: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket with optional prefix filtering.
        
        Args:
            prefix: S3 key prefix to filter by
            bucket_type: Type of bucket to list
            max_keys: Maximum number of objects to return
            
        Returns:
            List of object information dictionaries
        """
        try:
            target_bucket = self.buckets.get(bucket_type)
            if not target_bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            objects: List[Dict[str, Any]] = []
            continuation_token = None

            # Handle pagination for large listings
            while True:
                list_kwargs = {
                    "Bucket": target_bucket,
                    "Prefix": prefix,
                    "MaxKeys": max_keys
                }
                
                if continuation_token:
                    list_kwargs["ContinuationToken"] = continuation_token

                response = self.s3_client.list_objects_v2(**list_kwargs)

                # Process objects in current page
                for s3_object in response.get("Contents", []):
                    object_metadata = {}
                    
                    # Fetch detailed metadata for smaller listings
                    if max_keys <= 1000:
                        try:
                            head_response = self.s3_client.head_object(
                                Bucket=target_bucket, Key=s3_object["Key"]
                            )
                            object_metadata = head_response.get("Metadata", {})
                        except Exception:
                            pass

                    # Normalize timestamp format
                    last_modified = s3_object["LastModified"]
                    if isinstance(last_modified, datetime):
                        last_modified_iso = last_modified.astimezone(timezone.utc).isoformat()
                    else:
                        last_modified_iso = str(last_modified)

                    objects.append({
                        "key": s3_object["Key"],
                        "size_bytes": int(s3_object["Size"]),
                        "last_modified": last_modified_iso,
                        "etag": s3_object.get("ETag", "").strip('"'),
                        "metadata": object_metadata,
                        "storage_class": s3_object.get("StorageClass", "STANDARD"),
                    })

                # Check for more pages
                if not response.get("IsTruncated"):
                    break
                continuation_token = response.get("NextContinuationToken")

            # Sort by modification time (newest first)
            return sorted(objects, key=lambda x: x["last_modified"], reverse=True)

        except Exception as e:
            print(f"Failed to list objects: {e}")
            return []

    def list_experiment_artifacts(self, experiment_id: str, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all artifacts associated with a specific experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            artifact_type: Optional filter by artifact type
            
        Returns:
            List of experiment artifact dictionaries
        """
        try:
            artifacts: List[Dict[str, Any]] = []

            # List checkpoints if requested
            if artifact_type in (None, "checkpoints"):
                checkpoint_prefix = f"{self.prefixes.get('checkpoints', 'checkpoints/')}{experiment_id}/"
                for artifact in self.list_objects(checkpoint_prefix, "artifacts"):
                    artifact["artifact_type"] = "checkpoint"
                    artifacts.append(artifact)

            # List models if requested
            if artifact_type in (None, "models"):
                for model_type in ("tasker", "builder"):
                    model_prefix = f"{self.prefixes.get('models', 'models/')}{model_type}/{experiment_id}/"
                    
                    # Use models bucket if available, fallback to artifacts
                    bucket_type = "models" if self.buckets.get("models") else "artifacts"
                    
                    for artifact in self.list_objects(model_prefix, bucket_type):
                        artifact["artifact_type"] = "model"
                        artifact["model_type"] = model_type
                        artifacts.append(artifact)

            return sorted(artifacts, key=lambda x: x["last_modified"], reverse=True)

        except Exception as e:
            print(f"Failed to list experiment artifacts: {e}")
            return []

    # Object Management Operations

    def delete_object(self, s3_key: str, bucket_type: str = "artifacts") -> bool:
        """
        Delete single object from S3 bucket.
        
        Args:
            s3_key: S3 object key to delete
            bucket_type: Type of bucket containing the object
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            target_bucket = self.buckets.get(bucket_type)
            if not target_bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            self.s3_client.delete_object(Bucket=target_bucket, Key=s3_key)
            print(f"Deleted: s3://{target_bucket}/{s3_key}")
            return True

        except Exception as e:
            print(f"Failed to delete object: {e}")
            return False

    def delete_experiment_artifacts(self, experiment_id: str) -> int:
        """
        Delete all artifacts associated with an experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            Number of artifacts successfully deleted
        """
        try:
            artifacts = self.list_experiment_artifacts(experiment_id)
            deleted_count = 0

            for artifact in artifacts:
                # Determine appropriate bucket type
                if (artifact.get("artifact_type") == "model" and self.buckets.get("models")):
                    bucket_type = "models"
                else:
                    bucket_type = "artifacts"

                if self.delete_object(artifact["key"], bucket_type):
                    deleted_count += 1

            print(f"Deleted {deleted_count} artifacts for experiment {experiment_id}")
            return deleted_count

        except Exception as e:
            print(f"Failed to delete experiment artifacts: {e}")
            return 0

    # URL Generation and Access Control

    def create_presigned_url(
        self,
        s3_key: str,
        bucket_type: str = "artifacts",
        expiration_seconds: int = 3600,
        http_method: str = "get_object",
    ) -> Optional[str]:
        """
        Generate presigned URL for secure S3 object access.
        
        Args:
            s3_key: S3 object key
            bucket_type: Type of bucket containing the object
            expiration_seconds: URL expiration time in seconds
            http_method: HTTP method for the URL (get_object, put_object)
            
        Returns:
            Presigned URL string or None if generation failed
        """
        try:
            target_bucket = self.buckets.get(bucket_type)
            if not target_bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            presigned_url = self.s3_client.generate_presigned_url(
                ClientMethod=http_method,
                Params={"Bucket": target_bucket, "Key": s3_key},
                ExpiresIn=int(expiration_seconds),
            )

            print(f"Generated presigned URL ({http_method}, {expiration_seconds}s): {s3_key}")
            return presigned_url

        except Exception as e:
            print(f"Failed to create presigned URL: {e}")
            return None

    # Storage Analytics and Management

    def get_storage_usage(self, prefix: str = "", bucket_type: str = "artifacts") -> Dict[str, Any]:
        """
        Calculate storage usage statistics for bucket or prefix.
        
        Args:
            prefix: Optional S3 prefix to analyze
            bucket_type: Type of bucket to analyze
            
        Returns:
            Dictionary containing storage usage statistics
        """
        try:
            objects = self.list_objects(prefix, bucket_type, max_keys=10000)
            total_size = sum(obj["size_bytes"] for obj in objects)

            # Group by storage class
            storage_classes: Dict[str, Dict[str, int]] = {}
            for obj in objects:
                storage_class = obj.get("storage_class", "STANDARD")
                if storage_class not in storage_classes:
                    storage_classes[storage_class] = {"count": 0, "size_bytes": 0}
                
                storage_classes[storage_class]["count"] += 1
                storage_classes[storage_class]["size_bytes"] += obj["size_bytes"]

            return {
                "bucket_type": bucket_type,
                "prefix": prefix or "all",
                "object_count": len(objects),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
                "storage_classes": storage_classes,
            }

        except Exception as e:
            print(f"Failed to get storage usage: {e}")
            return {}