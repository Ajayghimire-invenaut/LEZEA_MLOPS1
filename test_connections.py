#!/usr/bin/env python3
"""Test all MLOps service connections (env-driven, no hardcoded names)."""

import os
import uuid
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

LINE = "=" * 50


def _info(msg: str) -> None:
    print(msg)


def _ok(service: str, extra: str = "") -> bool:
    print(f"✅ {service}: Connected successfully{(' — ' + extra) if extra else ''}")
    return True


def _warn(service: str, msg: str) -> bool:
    print(f"⚠️  {service}: {msg}")
    return False


def _err(service: str, msg: str) -> bool:
    print(f"❌ {service}: {msg}")
    return False


def test_postgresql() -> bool:
    """Optional: PostgreSQL (skip if driver missing or not configured)."""
    try:
        import psycopg2  # type: ignore
    except Exception:
        return _warn("PostgreSQL", "psycopg2 not installed (skipping — using SQLite is fine)")

    host = os.getenv("PGHOST", "localhost")
    db = os.getenv("PGDATABASE", "mlflow_db")
    user = os.getenv("PGUSER", "mlflow_user")
    pw = os.getenv("PGPASSWORD", "mlflow_password123")

    try:
        conn = psycopg2.connect(host=host, database=db, user=user, password=pw)
        conn.close()
        return _ok("PostgreSQL", f"{host}/{db}")
    except Exception as e:
        return _err("PostgreSQL", str(e))


def test_mongodb() -> bool:
    """MongoDB Atlas using env: MONGO_CONNECTION_STRING, MONGO_DATABASE."""
    uri = os.getenv("MONGO_CONNECTION_STRING", "")
    dbname = os.getenv("MONGO_DATABASE", "")

    if not uri or not dbname:
        return _warn("MongoDB", "MONGO_CONNECTION_STRING or MONGO_DATABASE not set (skipping)")

    try:
        from pymongo import MongoClient  # type: ignore
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")  # quick connectivity check

        # tiny write/delete to verify auth on target DB
        db = client[dbname]
        coll = db["_doctor"]
        token = uuid.uuid4().hex
        coll.insert_one({"_id": token, "ok": True})
        coll.delete_one({"_id": token})
        client.close()
        return _ok("MongoDB", f"db={dbname}")
    except Exception as e:
        return _err("MongoDB", str(e))


def _parse_s3_from_artifact_root(artifact_root: str) -> Tuple[str, str]:
    """
    Given s3://bucket/prefix... -> (bucket, prefix)
    If no prefix, returns ("bucket", "").
    """
    if not artifact_root.startswith("s3://"):
        return "", ""
    path = artifact_root[5:]
    if "/" in path:
        bucket, prefix = path.split("/", 1)
        return bucket, prefix
    return path, ""


def test_s3() -> bool:
    """S3 connectivity & minimal permissions using env:
       - MLFLOW_ARTIFACT_ROOT (to derive artifact bucket & prefix)
       - DATA_BUCKET (optional, for datasets)
       Also relies on standard AWS envs for credentials/region.
    """
    try:
        import boto3  # type: ignore
        from botocore.exceptions import ClientError  # type: ignore
    except Exception:
        return _err("S3", "boto3 not installed")

    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "")
    data_bucket = os.getenv("DATA_BUCKET", "")

    if not artifact_root.startswith("s3://"):
        return _warn("S3", "MLFLOW_ARTIFACT_ROOT not set to s3://... (skipping write test)")

    art_bucket, art_prefix = _parse_s3_from_artifact_root(artifact_root)
    required: List[str] = [b for b in [art_bucket, data_bucket] if b]
    # dedupe
    required = list(dict.fromkeys(required))

    try:
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
        # Check each required bucket exists (HeadBucket is cheapest)
        missing = []
        for b in required:
            try:
                s3.head_bucket(Bucket=b)
            except ClientError:
                missing.append(b)
        if missing:
            return _err("S3", f"Missing buckets: {missing}")

        # Minimal write/delete check in artifact bucket to validate Put/DeleteObject
        key = (art_prefix.rstrip("/") + "/" if art_prefix else "") + f"sanity/{uuid.uuid4().hex}.txt"
        s3.put_object(Bucket=art_bucket, Key=key, Body=b"ok")
        s3.delete_object(Bucket=art_bucket, Key=key)

        extra = f"artifact_bucket={art_bucket}"
        if data_bucket:
            extra += f", data_bucket={data_bucket}"
        return _ok("S3", extra)
    except Exception as e:
        return _err("S3", str(e))


def test_mlflow() -> bool:
    """MLflow connectivity using env:
       - MLFLOW_TRACKING_URI
       - MLFLOW_EXPERIMENT_NAME (optional, default: 'lezea_experiments')
       - MLFLOW_ARTIFACT_ROOT (used when creating experiment)
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return _warn("MLflow", "MLFLOW_TRACKING_URI not set (skipping)")

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "lezea_experiments")
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT")  # may be None; ok for set_experiment

    try:
        import mlflow  # type: ignore

        mlflow.set_tracking_uri(tracking_uri)

        # Ensure experiment exists and points to artifact root (if provided)
        existing = mlflow.get_experiment_by_name(exp_name)
        if existing is None:
            # Create with artifact root if available (S3)
            exp_id = mlflow.create_experiment(exp_name, artifact_location=artifact_root)
        else:
            exp_id = existing.experiment_id

        # Tiny run to verify we can talk to the server
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name="connection_test", tags={"checker": "mlops-doctor"}):
            mlflow.log_param("ping", "ok")

        return _ok("MLflow", f"uri={tracking_uri}, exp={exp_name}")
    except Exception as e:
        # Even if experiment already exists or server responds oddly, report the exception
        return _err("MLflow", str(e))


if __name__ == "__main__":
    print("🧪 Testing MLOps service connections...")
    print(LINE)

    results = [
        test_postgresql(),   # safe to skip on SQLite
        test_mongodb(),
        test_s3(),
        test_mlflow(),
    ]

    print("\n" + LINE)
    if all(results):
        print("🎉 ALL SERVICES CONNECTED SUCCESSFULLY!")
        print("Ready to start building the MLOps system.")
    else:
        print("⚠️  Some services failed. Fix the issues above.")
