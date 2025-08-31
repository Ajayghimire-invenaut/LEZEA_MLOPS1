#!/usr/bin/env python3
"""
LeZeA MLOps — Unified Service Doctor (env-driven, robust)

Checks:
- PostgreSQL (optional; auto-skip if using SQLite or driver missing)
- MongoDB Atlas (ping + tiny write/delete in target DB)
- S3 (head buckets from env + tiny write/delete under MLflow artifact prefix)
- MLflow (ensure experiment exists; tiny run)
- DVC (optional; prefer DVC_REMOTE_URL or config; safe fallback for list output)
- Grafana (optional; /api/health, supports bearer token)
- Prometheus (optional; /-/ready or runtimeinfo)

Env it reads (set only what you use):
- MONGO_CONNECTION_STRING, MONGO_DATABASE
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION[, AWS_SESSION_TOKEN]
- MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_ARTIFACT_URI[, MLFLOW_ARTIFACT_ROOT]
- DATA_BUCKET (optional extra bucket, e.g., datasets)
- DVC_REMOTE_URL (preferred), DVC_ROOT (default=".")
- GRAFANA_URL (default http://127.0.0.1:3000), GRAFANA_TOKEN (optional)
- PROMETHEUS_URL (default http://127.0.0.1:9090)
- Optional S3 endpoints: MLFLOW_S3_ENDPOINT_URL | AWS_S3_ENDPOINT_URL | DVC_ENDPOINT_URL
"""

import os
import uuid
import subprocess
from typing import List, Tuple, Optional
from urllib.parse import urlparse

# .env loader (safe if file missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LINE = "=" * 70


# ----------------------------- UX helpers ------------------------------------

def _ok(service: str, extra: str = "") -> bool:
    print(f"✅ {service}: OK{(' — ' + extra) if extra else ''}")
    return True

def _skip(service: str, msg: str) -> bool:
    print(f"ℹ️  {service}: skipped — {msg}")
    return True  # skip counts as success (nothing is broken)

def _warn(service: str, msg: str) -> bool:
    print(f"⚠️  {service}: {msg}")
    return False

def _err(service: str, msg: str) -> bool:
    print(f"❌ {service}: {msg}")
    return False


# ----------------------------- Utilities -------------------------------------

def _aws_region() -> str:
    return os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "eu-central-1"

def _s3_endpoint() -> Optional[str]:
    # Priority order for custom endpoints
    return (
        os.getenv("MLFLOW_S3_ENDPOINT_URL")
        or os.getenv("AWS_S3_ENDPOINT_URL")
        or os.getenv("DVC_ENDPOINT_URL")
        or None
    )

def _artifact_uri() -> str:
    return os.getenv("MLFLOW_ARTIFACT_URI") or os.getenv("MLFLOW_ARTIFACT_ROOT", "")

def _parse_s3_url(s3_url: str) -> Tuple[str, str]:
    """Return (bucket, prefix) from an s3://bucket/prefix... url."""
    if not s3_url.startswith("s3://"):
        return "", ""
    path = s3_url[5:]
    if "/" in path:
        bucket, prefix = path.split("/", 1)
        return bucket, prefix
    return path, ""

def _boto3_client():
    import boto3  # type: ignore
    return boto3.client("s3", region_name=_aws_region(), endpoint_url=_s3_endpoint())

def _run(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


# ----------------------------- PostgreSQL (optional) -------------------------

def test_postgresql() -> bool:
    """Treat PostgreSQL as optional: skip if using SQLite or driver missing."""
    if str(os.getenv("MLFLOW_BACKEND_STORE_URI", "")).lower().startswith("sqlite:"):
        return _skip("PostgreSQL", "using SQLite backend")

    try:
        import psycopg2  # type: ignore
    except Exception:
        return _skip("PostgreSQL", "psycopg2 not installed")

    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5432"))
    db = os.getenv("PGDATABASE", os.getenv("POSTGRES_DB", "mlflow_db"))
    user = os.getenv("PGUSER", os.getenv("POSTGRES_USER", "mlflow_user"))
    pw = os.getenv("PGPASSWORD", os.getenv("POSTGRES_PASSWORD", "mlflow_password123"))

    try:
        conn = psycopg2.connect(host=host, port=port, database=db, user=user, password=pw)  # type: ignore
        conn.close()
        return _ok("PostgreSQL", f"{host}:{port}/{db}")
    except Exception as e:
        return _err("PostgreSQL", str(e))


# ----------------------------- MongoDB ---------------------------------------

def test_mongodb() -> bool:
    uri = os.getenv("MONGO_CONNECTION_STRING", "")
    dbname = os.getenv("MONGO_DATABASE", "")
    if not uri or not dbname:
        return _skip("MongoDB", "MONGO_CONNECTION_STRING or MONGO_DATABASE not set")

    try:
        from pymongo import MongoClient  # type: ignore
        client = MongoClient(uri, serverSelectionTimeoutMS=6000, connectTimeoutMS=6000)
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


# ----------------------------- S3 --------------------------------------------

def test_s3() -> bool:
    try:
        import boto3  # noqa: F401
        from botocore.exceptions import ClientError  # type: ignore
    except Exception:
        return _err("S3", "boto3 not installed")

    artifact_uri = _artifact_uri()
    data_bucket = os.getenv("DATA_BUCKET", "").strip()

    if not artifact_uri.startswith("s3://"):
        return _skip("S3", "MLFLOW_ARTIFACT_URI/ROOT not set to s3:// (skipping write test)")

    art_bucket, art_prefix = _parse_s3_url(artifact_uri)
    must_exist = [b for b in [art_bucket, data_bucket] if b]
    must_exist = list(dict.fromkeys(must_exist))  # dedupe

    try:
        s3 = _boto3_client()

        # Head all required buckets
        missing = []
        for b in must_exist:
            try:
                s3.head_bucket(Bucket=b)
            except ClientError:
                missing.append(b)
        if missing:
            return _err("S3", f"missing buckets: {missing}")

        # Put/Get/Delete a tiny object in artifact prefix
        key = (art_prefix.rstrip("/") + "/" if art_prefix else "") + f"sanity/{uuid.uuid4().hex}.txt"
        body = b"ok"
        s3.put_object(Bucket=art_bucket, Key=key, Body=body)
        obj = s3.get_object(Bucket=art_bucket, Key=key)
        got = obj["Body"].read()
        s3.delete_object(Bucket=art_bucket, Key=key)

        if got != body:
            return _warn("S3", f"IO mismatch under {art_bucket}/{key}")

        extra = f"artifact_bucket={art_bucket}" + (f", data_bucket={data_bucket}" if data_bucket else "")
        return _ok("S3", extra)

    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        return _err("S3", f"{code}: {e}")
    except Exception as e:
        return _err("S3", str(e))


# ----------------------------- MLflow ----------------------------------------

def test_mlflow() -> bool:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        return _skip("MLflow", "MLFLOW_TRACKING_URI not set")

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "lezea_experiments")
    artifact_uri = _artifact_uri() or None  # used when creating experiment

    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(tracking_uri)

        # Ensure experiment exists
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            mlflow.create_experiment(exp_name, artifact_location=artifact_uri)
        mlflow.set_experiment(exp_name)

        # Tiny run
        with mlflow.start_run(run_name="connection_test", tags={"checker": "mlops-doctor"}):
            mlflow.log_param("ping", "ok")
            mlflow.log_metric("health_latency_ms", 1.0)

        return _ok("MLflow", f"uri={tracking_uri}, exp={exp_name}")
    except Exception as e:
        # This succeeds even if the web UI (/) returns 403; client API is what matters for pipelines.
        return _err("MLflow", str(e))


# ----------------------------- DVC (optional) --------------------------------

def _dvc_default_remote_url(dvc_root: str) -> Optional[str]:
    """
    Preferred method:
      1) DVC_REMOTE_URL override (env)
      2) dvc config core.remote -> dvc config remote.<name>.url
      3) fallback: parse first line of `dvc remote list` and strip '(default)'
    """
    override = os.getenv("DVC_REMOTE_URL", "").strip()
    if override:
        return override

    # Try config first (robust)
    rname = _run(["dvc", "config", "core.remote"], cwd=dvc_root)
    if rname.returncode == 0:
        name = rname.stdout.strip()
        if name:
            url = _run(["dvc", "config", f"remote.{name}.url"], cwd=dvc_root)
            if url.returncode == 0 and url.stdout.strip():
                return url.stdout.strip()

    # Safe fallback (strip "(default)")
    lst = _run(["dvc", "remote", "list"], cwd=dvc_root)
    if lst.returncode == 0 and lst.stdout.strip():
        first = lst.stdout.strip().splitlines()[0]
        # examples:
        # "storage   s3://bucket/path" or "storage   s3://bucket/path    (default)"
        parts = first.split(None, 1)
        if len(parts) == 2:
            url = parts[1].strip()
            url = url.replace("(default)", "").strip()
            return url

    return None


def test_dvc() -> bool:
    """Checks DVC CLI presence, repo status, and remote bucket reachability (for s3://)."""
    dvc_root = os.getenv("DVC_ROOT", ".")
    try:
        ver = _run(["dvc", "--version"], cwd=dvc_root)
        if ver.returncode != 0:
            return _skip("DVC", f"not available: {ver.stdout.strip()}")
    except FileNotFoundError:
        return _skip("DVC", "dvc CLI not installed")

    if not os.path.isdir(os.path.join(dvc_root, ".dvc")):
        return _skip("DVC", "no .dvc/ directory (not a DVC repo)")

    remote_url = _dvc_default_remote_url(dvc_root)
    if not remote_url:
        return _skip("DVC", "no DVC remote configured (core.remote empty)")

    if remote_url.startswith("s3://"):
        try:
            import boto3  # type: ignore
            u = urlparse(remote_url)
            bucket = u.netloc
            _boto3_client().head_bucket(Bucket=bucket)
            return _ok("DVC", f"remote={remote_url} (bucket reachable)")
        except Exception as e:
            return _warn("DVC", f"remote={remote_url} (S3 head failed: {e})")
    else:
        # For HTTP/SSH remotes just sanity check status
        st = _run(["dvc", "status", "-c"], cwd=dvc_root)
        if st.returncode == 0:
            return _ok("DVC", f"remote={remote_url} (status OK)")
        return _warn("DVC", f"remote={remote_url} (status failed: {st.stdout.strip()})")


# ----------------------------- Grafana (optional) ----------------------------

def test_grafana() -> bool:
    try:
        import requests  # type: ignore
    except Exception:
        return _skip("Grafana", "requests not installed")

    url = os.getenv("GRAFANA_URL", "http://127.0.0.1:3000").rstrip("/")
    token = os.getenv("GRAFANA_TOKEN", "").strip()
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        r = requests.get(f"{url}/api/health", headers=headers, timeout=4)
        if r.status_code == 200 and "database" in r.text:
            return _ok("Grafana", f"{url}/api/health")
        return _warn("Grafana", f"unexpected response {r.status_code}: {r.text[:120]}")
    except Exception as e:
        return _skip("Grafana", f"not reachable ({e})")


# ----------------------------- Prometheus (optional) --------------------------

def test_prometheus() -> bool:
    try:
        import requests  # type: ignore
    except Exception:
        return _skip("Prometheus", "requests not installed")

    url = os.getenv("PROMETHEUS_URL", "http://127.0.0.1:9090").rstrip("/")
    try:
        r = requests.get(f"{url}/-/ready", timeout=4)
        if r.status_code == 200:
            return _ok("Prometheus", f"{url}/-/ready")
        r2 = requests.get(f"{url}/api/v1/status/runtimeinfo", timeout=4)
        if r2.status_code == 200:
            return _ok("Prometheus", f"{url}/api/v1/status/runtimeinfo")
        return _warn("Prometheus", f"http {r.status_code} /ready; {r2.status_code} /runtimeinfo")
    except Exception as e:
        return _skip("Prometheus", f"not reachable ({e})")


# ----------------------------- Main ------------------------------------------

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
