# lezea_mlops/backends/s3_backend.py
# S3 Backend for LeZeA MLOps — hardened

from __future__ import annotations

import os
import json
import hashlib
import mimetypes
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
    from botocore.config import Config as BotoCoreConfig
    from boto3.s3.transfer import TransferConfig as S3TransferConfig
    S3_AVAILABLE = True
except ImportError:  # pragma: no cover
    boto3 = None
    BotoCoreConfig = object  # type: ignore
    S3TransferConfig = object  # type: ignore
    ClientError = NoCredentialsError = BotoCoreError = Exception  # type: ignore
    S3_AVAILABLE = False


def _guess_content_type(path: str) -> str:
    # Better than hand-rolled map; handles .tar.gz, etc.
    ctype, _ = mimetypes.guess_type(path)
    return ctype or "application/octet-stream"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


class S3Backend:
    """
    AWS S3 (or S3-compatible) backend for artifacts & models.

    Features:
    - Proper multipart via boto3 TransferConfig
    - Optional SSE-S3 / SSE-KMS
    - MinIO / custom endpoint support
    - Integrity checks (metadata SHA256)
    - Presigned GET/PUT URLs
    - Health ping()
    """

    def __init__(self, config):
        if not S3_AVAILABLE:
            raise RuntimeError("boto3 not available. Install with: pip install boto3")

        self.config = config
        self.s3_config = config.get_s3_config()

        # Read settings
        self.region: str = self.s3_config.get("region", "us-east-1")
        self.endpoint_url: Optional[str] = self.s3_config.get("endpoint_url")  # e.g., MinIO
        self.use_accelerate: bool = bool(self.s3_config.get("accelerate", False))
        self.sse: Optional[str] = self.s3_config.get("sse")  # "AES256" or "aws:kms"
        self.kms_key_id: Optional[str] = self.s3_config.get("kms_key_id")
        self.storage_class: Optional[str] = self.s3_config.get("storage_class")  # e.g., "STANDARD_IA"

        # Buckets/prefixes
        self.buckets: Dict[str, str] = self.s3_config.get("buckets", {})
        self.prefixes: Dict[str, str] = self.s3_config.get("prefixes", {})

        # Build client (let AWS default credential chain work; don't force keys)
        session_kwargs = {}
        if self.s3_config.get("aws_access_key_id") and self.s3_config.get("aws_secret_access_key"):
            session_kwargs.update(
                aws_access_key_id=self.s3_config["aws_access_key_id"],
                aws_secret_access_key=self.s3_config["aws_secret_access_key"],
                aws_session_token=self.s3_config.get("aws_session_token"),
            )

        self._boto_cfg = BotoCoreConfig(
            region_name=self.region,
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=int(self.s3_config.get("max_pool_connections", 50)),
            signature_version=self.s3_config.get("signature_version", "s3v4"),
        )
        self._xfer_cfg = S3TransferConfig(
            multipart_threshold=int(self.s3_config.get("multipart_threshold", 64 * 1024 * 1024)),
            multipart_chunksize=int(self.s3_config.get("multipart_chunksize", 16 * 1024 * 1024)),
            max_concurrency=int(self.s3_config.get("max_concurrency", 10)),
            use_threads=True,
        )

        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                config=self._boto_cfg,
                **session_kwargs,
            )
            # Optional: enable accelerate on URLs (client must still address the same bucket)
            if self.use_accelerate:
                try:
                    for b in set(self.buckets.values()):
                        if b:
                            self.s3_client.put_bucket_accelerate_configuration(
                                Bucket=b, AccelerateConfiguration={"Status": "Enabled"}
                            )
                except Exception:
                    # Not all providers/buckets support this — ignore
                    pass
        except NoCredentialsError as e:
            raise ValueError("No AWS credentials found (env/role/credentials file).") from e
        except Exception as e:
            raise ConnectionError(f"Failed to initialize S3 client: {e}") from e

        # Verify primary bucket access early
        self._verify_bucket_access()

        self.available = True
        print(f"✅ S3 backend connected (region={self.region}, endpoint={self.endpoint_url or 'aws'})")

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    def ping(self) -> bool:
        """Used by tracker health check."""
        try:
            artifacts_bucket = self.buckets.get("artifacts")
            if artifacts_bucket:
                self.s3_client.head_bucket(Bucket=artifacts_bucket)
                return True
            # If no artifacts bucket configured, just try STS or list_buckets (may be restricted)
            self.s3_client.list_buckets()
            return True
        except Exception:
            return False

    def _verify_bucket_access(self) -> None:
        try:
            # Validate artifacts bucket (primary)
            artifacts_bucket = self.buckets.get("artifacts")
            if artifacts_bucket:
                self.s3_client.head_bucket(Bucket=artifacts_bucket)
                print(f"✅ Artifacts bucket ok: {artifacts_bucket}")

            # Check others gently
            for k, b in self.buckets.items():
                if not b or k == "artifacts":
                    continue
                try:
                    self.s3_client.head_bucket(Bucket=b)
                    print(f"✅ {k} bucket ok: {b}")
                except ClientError as e:
                    print(f"⚠️ {k} bucket not accessible ({b}): {e.response.get('Error', {}).get('Code')}")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            raise ConnectionError(f"S3 bucket access error: {code}") from e

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _meta_for(self, file_path: Path, extra: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        meta: Dict[str, str] = {
            "original_filename": file_path.name,
            "file_size_bytes": str(file_path.stat().st_size if file_path.exists() else 0),
            "file_hash_sha256": _sha256(file_path) if file_path.exists() else "",
            "upload_timestamp": datetime.now(timezone.utc).isoformat(),
            "uploaded_by": "lezea_mlops",
        }
        if extra:
            for k, v in extra.items():
                meta[f"custom_{k}"] = str(v)
        return meta

    def _extra_args(self, content_type: str, metadata: Dict[str, str]) -> Dict[str, Any]:
        args: Dict[str, Any] = {"ContentType": content_type, "Metadata": metadata}
        if self.storage_class:
            args["StorageClass"] = self.storage_class
        if self.sse:
            args["ServerSideEncryption"] = self.sse
            if self.sse == "aws:kms" and self.kms_key_id:
                args["SSEKMSKeyId"] = self.kms_key_id
        return args

    # ------------------------------------------------------------------ #
    # Upload APIs used by the tracker
    # ------------------------------------------------------------------ #
    def upload_checkpoint(
        self,
        local_path: str,
        experiment_id: str,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Optional[str]:
        """Upload a training checkpoint into artifacts bucket."""
        try:
            p = Path(local_path)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {local_path}")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            fname = p.name
            prefix = self.prefixes.get("checkpoints", "checkpoints/")
            if step is not None:
                key = f"{prefix}{experiment_id}/step_{int(step)}/{ts}_{fname}"
            else:
                key = f"{prefix}{experiment_id}/{ts}/{fname}"

            meta = self._meta_for(p, {"experiment_id": experiment_id, "checkpoint_step": step or -1,
                                      "checkpoint_type": "training_checkpoint", **(metadata or {})})
            bucket = self.buckets.get("artifacts")
            if not bucket:
                raise ValueError("Artifacts bucket not configured")

            self._upload_file(bucket, key, p, meta, progress_cb)
            print(f"📤 Uploaded checkpoint: {fname} → s3://{bucket}/{key}")
            return key
        except Exception as e:
            print(f"❌ Failed to upload checkpoint: {e}")
            return None

    def upload_final_model(
        self,
        local_path: str,
        experiment_id: str,
        model_type: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Optional[str]:
        """Upload a final model into models (or artifacts) bucket."""
        try:
            p = Path(local_path)
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {local_path}")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            fname = p.name
            prefix = self.prefixes.get("models", "models/")
            key = f"{prefix}{model_type}/{experiment_id}/{model_name}_{ts}/{fname}"

            meta = self._meta_for(p, {"experiment_id": experiment_id, "model_type": model_type,
                                      "model_name": model_name, "model_category": "final_model", **(metadata or {})})
            bucket = self.buckets.get("models") or self.buckets.get("artifacts")
            if not bucket:
                raise ValueError("Models (or artifacts) bucket not configured")

            self._upload_file(bucket, key, p, meta, progress_cb)
            print(f"🏆 Uploaded final model: {model_name} → s3://{bucket}/{key}")
            return key
        except Exception as e:
            print(f"❌ Failed to upload final model: {e}")
            return None

    # Core uploader with TransferConfig + progress
    def _upload_file(
        self,
        bucket: str,
        key: str,
        path: Path,
        metadata: Dict[str, str],
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> None:
        class _CB:
            def __init__(self, cb): self.cb, self.n = cb, 0
            def __call__(self, bytes_amount):
                if self.cb:
                    self.n += bytes_amount
                    try: self.cb(self.n)
                    except Exception: pass

        extra_args = self._extra_args(_guess_content_type(str(path)), metadata)
        self.s3_client.upload_file(
            Filename=str(path),
            Bucket=bucket,
            Key=key,
            ExtraArgs=extra_args,
            Callback=_CB(progress_cb),
            Config=self._xfer_cfg,
        )

    # ------------------------------------------------------------------ #
    # Generic artifacts
    # ------------------------------------------------------------------ #
    def upload_artifact(
        self,
        local_path: str,
        artifact_key: str,
        bucket_type: str = "artifacts",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        try:
            p = Path(local_path)
            if not p.exists():
                raise FileNotFoundError(f"Artifact not found: {local_path}")
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            meta = self._meta_for(p, {"artifact_type": "general", **(metadata or {})})
            self._upload_file(bucket, artifact_key, p, meta, None)
            print(f"📎 Uploaded artifact: {p.name} → s3://{bucket}/{artifact_key}")
            return artifact_key
        except Exception as e:
            print(f"❌ Failed to upload artifact: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Download / list / delete
    # ------------------------------------------------------------------ #
    def download_file(
        self,
        s3_key: str,
        local_path: str,
        bucket_type: str = "artifacts",
        verify_hash: bool = True,
    ) -> bool:
        try:
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            dirpath = os.path.dirname(local_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

            self.s3_client.download_file(bucket, s3_key, local_path)

            if verify_hash:
                try:
                    head = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
                    stored = head.get("Metadata", {}).get("file_hash_sha256")
                    if stored:
                        local_hash = _sha256(Path(local_path))
                        if stored != local_hash:
                            print(f"⚠️ Integrity mismatch for {s3_key}")
                            return False
                        else:
                            print(f"✅ Integrity OK for {s3_key}")
                except Exception:
                    # Non-fatal
                    pass

            print(f"📥 Downloaded: s3://{bucket}/{s3_key} → {local_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to download file: {e}")
            return False

    def list_objects(
        self,
        prefix: str = "",
        bucket_type: str = "artifacts",
        max_keys: int = 1000,
    ) -> List[Dict[str, Any]]:
        try:
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            # paginate
            objects: List[Dict[str, Any]] = []
            token = None
            while True:
                kw = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": max_keys}
                if token:
                    kw["ContinuationToken"] = token
                resp = self.s3_client.list_objects_v2(**kw)
                for obj in resp.get("Contents", []):
                    # head might be slow; only fetch metadata if small listing
                    metadata = {}
                    if max_keys <= 1000:
                        try:
                            head = self.s3_client.head_object(Bucket=bucket, Key=obj["Key"])
                            metadata = head.get("Metadata", {})
                        except Exception:
                            pass
                    lm = obj["LastModified"]
                    # normalize to ISO string UTC
                    if isinstance(lm, datetime):
                        lm_iso = lm.astimezone(timezone.utc).isoformat()
                    else:
                        lm_iso = str(lm)
                    objects.append(
                        {
                            "key": obj["Key"],
                            "size_bytes": int(obj["Size"]),
                            "last_modified": lm_iso,
                            "etag": obj.get("ETag", "").strip('"'),
                            "metadata": metadata,
                            "storage_class": obj.get("StorageClass", "STANDARD"),
                        }
                    )
                if not resp.get("IsTruncated"):
                    break
                token = resp.get("NextContinuationToken")
            return sorted(objects, key=lambda x: x["last_modified"], reverse=True)
        except Exception as e:
            print(f"❌ Failed to list objects: {e}")
            return []

    def list_experiment_artifacts(self, experiment_id: str, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            out: List[Dict[str, Any]] = []
            if artifact_type in (None, "checkpoints"):
                cp_prefix = f"{self.prefixes.get('checkpoints', 'checkpoints/')}{experiment_id}/"
                for it in self.list_objects(cp_prefix, "artifacts"):
                    it["artifact_type"] = "checkpoint"
                    out.append(it)
            if artifact_type in (None, "models"):
                for mt in ("tasker", "builder"):
                    model_prefix = f"{self.prefixes.get('models', 'models/')}{mt}/{experiment_id}/"
                    # models bucket may be absent; fallback to artifacts
                    bucket_type = "models" if self.buckets.get("models") else "artifacts"
                    for it in self.list_objects(model_prefix, bucket_type):
                        it["artifact_type"] = "model"
                        it["model_type"] = mt
                        out.append(it)
            return sorted(out, key=lambda x: x["last_modified"], reverse=True)
        except Exception as e:
            print(f"❌ Failed to list experiment artifacts: {e}")
            return []

    def delete_object(self, s3_key: str, bucket_type: str = "artifacts") -> bool:
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
        try:
            items = self.list_experiment_artifacts(experiment_id)
            n = 0
            for it in items:
                btype = "models" if it.get("artifact_type") == "model" and self.buckets.get("models") else "artifacts"
                if self.delete_object(it["key"], btype):
                    n += 1
            print(f"🗑️ Deleted {n} artifacts for experiment {experiment_id}")
            return n
        except Exception as e:
            print(f"❌ Failed to delete experiment artifacts: {e}")
            return 0

    # ------------------------------------------------------------------ #
    # URLs & stats
    # ------------------------------------------------------------------ #
    def create_presigned_url(
        self,
        s3_key: str,
        bucket_type: str = "artifacts",
        expiration: int = 3600,
        method: str = "get_object",
    ) -> Optional[str]:
        try:
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")
            url = self.s3_client.generate_presigned_url(
                ClientMethod=method,
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=int(expiration),
            )
            print(f"🔗 Presigned URL ({method}, {expiration}s): {s3_key}")
            return url
        except Exception as e:
            print(f"❌ Failed to create presigned URL: {e}")
            return None

    def get_storage_usage(self, prefix: str = "", bucket_type: str = "artifacts") -> Dict[str, Any]:
        try:
            objs = self.list_objects(prefix, bucket_type, max_keys=10000)
            total = sum(it["size_bytes"] for it in objs)
            by_class: Dict[str, Dict[str, int]] = {}
            for it in objs:
                sc = it.get("storage_class", "STANDARD")
                by_class.setdefault(sc, {"count": 0, "size_bytes": 0})
                by_class[sc]["count"] += 1
                by_class[sc]["size_bytes"] += it["size_bytes"]
            return {
                "bucket_type": bucket_type,
                "prefix": prefix or "all",
                "object_count": len(objs),
                "total_size_bytes": total,
                "total_size_mb": round(total / (1024 * 1024), 2),
                "total_size_gb": round(total / (1024 * 1024 * 1024), 2),
                "storage_classes": by_class,
            }
        except Exception as e:
            print(f"❌ Failed to get storage usage: {e}")
            return {}

    # ------------------------------------------------------------------ #
    # Sync helpers
    # ------------------------------------------------------------------ #
    def sync_directory(
        self,
        local_dir: str,
        s3_prefix: str,
        bucket_type: str = "artifacts",
        delete_missing: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        stats = {"uploaded": 0, "skipped": 0, "deleted": 0, "errors": 0}
        try:
            root = Path(local_dir)
            if not root.exists():
                raise FileNotFoundError(f"Local directory not found: {local_dir}")
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                raise ValueError(f"Bucket type '{bucket_type}' not configured")

            # Upload phase
            for fp in root.rglob("*"):
                if not fp.is_file():
                    continue
                rel = fp.relative_to(root).as_posix()
                key = f"{s3_prefix.rstrip('/')}/{rel}"
                try:
                    # quick skip check via metadata hash
                    head = self.s3_client.head_object(Bucket=bucket, Key=key)
                    stored = head.get("Metadata", {}).get("file_hash_sha256")
                    if stored and stored == _sha256(fp):
                        stats["skipped"] += 1
                        continue
                except ClientError:
                    pass  # key not found

                if dry_run:
                    print(f"[DRY] Upload {fp} → s3://{bucket}/{key}")
                    stats["uploaded"] += 1
                else:
                    if self.upload_artifact(str(fp), key, bucket_type):
                        stats["uploaded"] += 1
                    else:
                        stats["errors"] += 1

            # Delete phase
            if delete_missing:
                remote = self.list_objects(s3_prefix, bucket_type, max_keys=10000)
                remote_keys = {it["key"] for it in remote}
                local_keys = {f"{s3_prefix.rstrip('/')}/{fp.relative_to(root).as_posix()}"
                              for fp in root.rglob("*") if fp.is_file()}
                to_del = remote_keys - local_keys
                for key in to_del:
                    if dry_run:
                        print(f"[DRY] Delete s3://{bucket}/{key}")
                        stats["deleted"] += 1
                    else:
                        if self.delete_object(key, bucket_type):
                            stats["deleted"] += 1
                        else:
                            stats["errors"] += 1

            print(f"📁 {'Would sync' if dry_run else 'Synced'}: {stats}")
            return stats
        except Exception as e:
            print(f"❌ Failed to sync directory: {e}")
            stats["errors"] += 1
            return stats

    # ------------------------------------------------------------------ #
    # Info / maintenance
    # ------------------------------------------------------------------ #
    def get_bucket_info(self, bucket_type: str = "artifacts") -> Dict[str, Any]:
        try:
            bucket = self.buckets.get(bucket_type)
            if not bucket:
                return {"error": f"Bucket type '{bucket_type}' not configured", "accessible": False}
            try:
                loc = self.s3_client.get_bucket_location(Bucket=bucket)
                region = loc.get("LocationConstraint") or "us-east-1"
            except Exception:
                region = self.region
            sample = self.list_objects("", bucket_type, max_keys=500)
            size = sum(o["size_bytes"] for o in sample)
            return {
                "bucket_name": bucket,
                "bucket_type": bucket_type,
                "region": region,
                "sample_object_count": len(sample),
                "sample_total_size_mb": round(size / (1024 * 1024), 2),
                "accessible": True,
            }
        except Exception as e:
            return {"bucket_type": bucket_type, "error": str(e), "accessible": False}

    def cleanup_temp_files(self, max_age_days: int = 7) -> int:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            prefix = self.prefixes.get("temp", "temp/")
            items = self.list_objects(prefix, "artifacts", max_keys=10000)
            n = 0
            for it in items:
                try:
                    lm = datetime.fromisoformat(it["last_modified"].replace("Z", "+00:00"))
                except Exception:
                    # best effort: skip unparsable dates
                    continue
                if lm < cutoff:
                    if self.delete_object(it["key"], "artifacts"):
                        n += 1
            print(f"🧹 Cleaned {n} temp objects older than {max_age_days}d")
            return n
        except Exception as e:
            print(f"❌ Failed to cleanup temp files: {e}")
            return 0
