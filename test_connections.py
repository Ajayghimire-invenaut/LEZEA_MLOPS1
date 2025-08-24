#!/usr/bin/env python3
"""
LeZeA MLOps — Unified Service Doctor (env-driven, no hardcoded names)

Checks:
- PostgreSQL (optional; auto-skip if using SQLite or driver missing)
- MongoDB Atlas (ping + tiny write/delete in target DB)
- S3 (head buckets from env + tiny write/delete under MLflow artifact prefix)
- MLflow (ensure experiment exists; tiny run)
- DVC (optional; detect remote from `dvc remote list` or DVC_REMOTE; smoke a remote check)
- Grafana (optional; /api/health, supports bearer token)
- Prometheus (optional; /-/ready or runtimeinfo)

Env it reads (set only what you use):
- MONGO_CONNECTION_STRING, MONGO_DATABASE
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
- MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_ARTIFACT_ROOT
- DATA_BUCKET (optional second bucket, e.g. datasets)
- DVC_REMOTE (optional override), DVC_ROOT (optional path; default=".")
- GRAFANA_URL (default http://127.0.0.1:3000), GRAFANA_TOKEN (optional)
- PROMETHEUS_URL (default http://127.0.0.1:9090)
"""

import os
import uuid
import subprocess
from typing import List, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()

LINE = "=" * 70


def _ok(service: str, extra: str = "") -> bool:
    print(f"✅ {service}: OK{(' — ' + extra) if extra else ''}")
    return True


def _skip(service: str, msg: str) -> bool:
    print(f"ℹ️  {service}: skipped — {msg}")
    return True  # skip counts as success


def _warn(service: str, msg: str) -> bool:
    print(f"⚠️  {service}: {msg}")
    return False


def _err(service: str, msg: str) -> bool:
    print(f"❌ {service}: {msg}")
    return False


# ----------------------------- PostgreSQL (optional)
def test_postgresql() -> bool:
    """Treat PostgreSQL as optional: if using SQLite, or psycopg2 missing, skip."""
    if str(os.getenv("MLFLOW_BACKEND_STORE_URI", "")).startswith("sqlite:"):
        return _skip("PostgreSQL", "using SQLite backend")

    try:
        import psycopg2  # type: ignore
    except Exception:
        return _skip("PostgreSQL", "psycopg2 not installed")

    host = os.getenv("PGHOST", "localhost")
    db = os.getenv("PGDATABASE", "mlflow_db")
    user = os.getenv("PGUSER", "mlflow_user")
    pw = os.getenv("PGPASSWORD", "mlflow_password123")

    try:
        conn = psycopg2.connect(host=host, database=db, user=user, password=pw)  # type: ignore
        conn.close()
        return _ok("PostgreSQL", f"{host}/{db}")
    except Exception as e:
        return _err("PostgreSQL", str(e))


# ----------------------------- MongoDB
def test_mongodb() -> bool:
    uri = os.getenv("MONGO_CONNECTION_STRING", "")
    dbname = os.getenv("MONGO_DATABASE", "")
    if not uri or not dbname:
        return _skip("MongoDB", "MONGO_CONNECTION_STRING or MONGO_DATABASE not set")

    try:
        from pymongo import MongoClient  # type: ignore
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")

        db = client[dbname]
        token = uuid.uuid4().hex
        coll = db["_doctor"]
        coll.insert_one({"_id": token, "ok": True})
        coll.delete_one({"_id": token})
        client.close()
        return _ok("MongoDB", f"db={dbname}")
    except Exception as e:
        return _err("MongoDB", str(e))


# ----------------------------- S3
def _parse_s3_from_artifact_root(artifact_root: str) -> Tuple[str, str]:
    if not artifact_root.startswith("s3://"):
        return "", ""
    path = artifact_root[5:]
    if "/" in path:
        bucket, prefix = path.split("/", 1)
        return bucket, prefix
    return path, ""


def test_s3() -> bool:
    try:
        import boto3  # type: ignore
        from botocore.exceptions import ClientError  # type: ignore
    except Exception:
        return _err("S3", "boto3 not installed")

    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "")
    data_bucket = os.getenv("DATA_BUCKET", "")

    if not artifact_root.startswith("s3://"):
        return _skip("S3", "MLFLOW_ARTIFACT_ROOT not set to s3:// (skipping write test)")

    art_bucket, art_prefix = _parse_s3_from_artifact_root(artifact_root)
    required: List[str] = [b for b in [art_bucket, data_bucket] if b]
    required = list(dict.fromkeys(required))  # dedupe

    try:
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
        # buckets exist?
        missing = []
        for b in required:
            try:
                s3.head_bucket(Bucket=b)
            except ClientError:
                missing.append(b)
        if missing:
            return _err("S3", f"missing buckets: {missing}")

        # write/delete sanity in artifact prefix
        key = (art_prefix.rstrip("/") + "/" if art_prefix else "") + f"sanity/{uuid.uuid4().hex}.txt"
        s3.put_object(Bucket=art_bucket, Key=key, Body=b"ok")
        s3.head_object(Bucket=art_bucket, Key=key)
        s3.delete_object(Bucket=art_bucket, Key=key)

        extra = f"artifact_bucket={art_bucket}" + (f", data_bucket={data_bucket}" if data_bucket else "")
        return _ok("S3", extra)
    except Exception as e:
        return _err("S3", str(e))


# ----------------------------- MLflow
def test_mlflow() -> bool:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return _skip("MLflow", "MLFLOW_TRACKING_URI not set")

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "lezea_experiments")
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT")

    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(tracking_uri)

        existing = mlflow.get_experiment_by_name(exp_name)
        if existing is None:
            mlflow.create_experiment(exp_name, artifact_location=artifact_root)
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name="connection_test", tags={"checker": "mlops-doctor"}):
            mlflow.log_param("ping", "ok")
            mlflow.log_metric("health_latency_ms", 1.0)

        return _ok("MLflow", f"uri={tracking_uri}, exp={exp_name}")
    except Exception as e:
        return _err("MLflow", str(e))


# ----------------------------- DVC (optional)
def _run(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_dvc() -> bool:
    """Checks DVC CLI presence, current repo status, and remote basic sanity.
       - Uses DVC_REMOTE if set; else parses `dvc remote list`.
       - If remote is s3://..., HEAD the bucket (via boto3).
       Skips cleanly if DVC not installed or no repo.
    """
    dvc_root = os.getenv("DVC_ROOT", ".")
    try:
        r = _run(["dvc", "--version"], cwd=dvc_root)
        if r.returncode != 0:
            return _skip("DVC", f"not available: {r.stdout.strip()}")
    except FileNotFoundError:
        return _skip("DVC", "dvc CLI not installed")

    # Is this a DVC repo?
    if not os.path.isdir(os.path.join(dvc_root, ".dvc")):
        return _skip("DVC", "no .dvc/ directory (not a DVC repo)")

    # Remote detection
    remote_url = os.getenv("DVC_REMOTE", "").strip()
    if not remote_url:
        rl = _run(["dvc", "remote", "list"], cwd=dvc_root)
        if rl.returncode != 0 or not rl.stdout.strip():
            return _skip("DVC", "no DVC remote configured")
        # Take first remote's URL (e.g. "storage  s3://bucket/prefix")
        remote_url = rl.stdout.strip().splitlines()[0].split(None, 1)[-1].strip()

    # Try a lightweight check
    if remote_url.startswith("s3://"):
        try:
            import boto3  # type: ignore
            from urllib.parse import urlparse
            u = urlparse(remote_url)
            bucket = u.netloc
            s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
            s3.head_bucket(Bucket=bucket)
            return _ok("DVC", f"remote={remote_url} (bucket reachable)")
        except Exception as e:
            return _warn("DVC", f"remote={remote_url} (S3 head failed: {e})")
    else:
        # for http/ssh remotes, just trust `remote list` and `status -c`
        st = _run(["dvc", "status", "-c"], cwd=dvc_root)
        if st.returncode == 0:
            return _ok("DVC", f"remote={remote_url} (status OK)")
        return _warn("DVC", f"remote={remote_url} (status failed: {st.stdout.strip()})")


# ----------------------------- Grafana (optional)
def test_grafana() -> bool:
    import requests  # type: ignore
    url = os.getenv("GRAFANA_URL", "http://127.0.0.1:3000").rstrip("/")
    token = os.getenv("GRAFANA_TOKEN", "").strip()
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(f"{url}/api/health", headers=headers, timeout=3)
        if r.status_code == 200 and "database" in r.text:
            return _ok("Grafana", f"{url}/api/health")
        return _warn("Grafana", f"unexpected response {r.status_code}: {r.text[:120]}")
    except Exception as e:
        return _skip("Grafana", f"not reachable ({e})")


# ----------------------------- Prometheus (optional)
def test_prometheus() -> bool:
    import requests  # type: ignore
    url = os.getenv("PROMETHEUS_URL", "http://127.0.0.1:9090").rstrip("/")
    try:
        r = requests.get(f"{url}/-/ready", timeout=3)
        if r.status_code == 200:
            return _ok("Prometheus", f"{url}/-/ready")
        # fallback to runtimeinfo
        r2 = requests.get(f"{url}/api/v1/status/runtimeinfo", timeout=3)
        if r2.status_code == 200:
            return _ok("Prometheus", f"{url}/api/v1/status/runtimeinfo")
        return _warn("Prometheus", f"http {r.status_code} /ready; {r2.status_code} /runtimeinfo")
    except Exception as e:
        return _skip("Prometheus", f"not reachable ({e})")


# ----------------------------- Main
if __name__ == "__main__":
    print("🧪 LeZeA MLOps — Service Doctor")
    print(LINE)

    checks = [
        test_postgresql,   # optional; treated as skip if SQLite
        test_mongodb,
        test_s3,
        test_mlflow,
        test_dvc,          # optional
        test_grafana,      # optional
        test_prometheus,   # optional
    ]

    results = [fn() for fn in checks]

    print("\n" + LINE)
    if all(results):
        print("🎉 ALL CHECKS GREEN — platform is healthy.")
    else:
        print("⚠️  Some checks did not pass. See messages above.")
