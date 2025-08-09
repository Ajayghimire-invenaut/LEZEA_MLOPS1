#!/usr/bin/env python3
# scripts/export_grafana.py
"""
Export PNGs from Grafana dashboards and log them to MLflow.

Requires:
  - requests
  - mlflow

Auth:
  - Provide a Grafana API token with "Viewer" (and renderer enabled) via --token or GRAFANA_API_TOKEN env var.

Examples:
  python scripts/export_grafana.py \
      --url https://grafana.example.com \
      --uid AbCdEfGhZ \
      --panels 2,4,7 \
      --hours 6 \
      --mlflow-run-id $MLFLOW_RUN_ID

Notes:
  - We auto-fetch the dashboard slug using /api/dashboards/uid/{uid}.
  - Time range defaults to now-6h .. now.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional

import requests
import mlflow


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _get_slug(base: str, token: str, uid: str) -> str:
    url = f"{base}/api/dashboards/uid/{uid}"
    r = requests.get(url, headers=_headers(token), timeout=30)
    r.raise_for_status()
    js = r.json()
    slug = js.get("meta", {}).get("slug") or js.get("dashboard", {}).get("title", "dashboard")
    # Fallback sanitize
    return str(slug).replace(" ", "-").lower()


def _render_panel_png(
    base: str,
    token: str,
    uid: str,
    slug: str,
    panel_id: int,
    t_from_ms: int,
    t_to_ms: int,
    org_id: int,
    width: int,
    height: int,
    theme: str = "light",
) -> bytes:
    """
    Use Grafana image renderer endpoint:
      GET /render/d-solo/{uid}/{slug}?panelId=...&from=...&to=...&width=...&height=...&orgId=1&theme=light
    """
    render_url = f"{base}/render/d-solo/{uid}/{slug}"
    params = {
        "panelId": panel_id,
        "from": t_from_ms,
        "to": t_to_ms,
        "width": width,
        "height": height,
        "orgId": org_id,
        "theme": theme,
    }
    r = requests.get(render_url, headers=_headers(token), params=params, timeout=60)
    r.raise_for_status()
    return r.content


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Export Grafana panels as PNG and log to MLflow.")
    ap.add_argument("--url", required=True, help="Base Grafana URL, e.g., https://grafana.example.com")
    ap.add_argument("--token", default=os.getenv("GRAFANA_API_TOKEN"), help="Grafana API token")
    ap.add_argument("--uid", required=True, help="Dashboard UID (single)")
    ap.add_argument("--panels", required=True, help="Comma-separated panel IDs, e.g., 2,4,7")
    ap.add_argument("--hours", type=int, default=6, help="Lookback window in hours (default: 6)")
    ap.add_argument("--from-ms", type=int, default=None, help="Override start time (epoch ms)")
    ap.add_argument("--to-ms", type=int, default=None, help="Override end time (epoch ms)")
    ap.add_argument("--org-id", type=int, default=1, help="Grafana org ID (default: 1)")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--outdir", default="exports/grafana")
    ap.add_argument("--mlflow-run-id", default=os.getenv("MLFLOW_RUN_ID"), help="Existing MLflow run to log into")
    ap.add_argument("--artifact-path", default="grafana", help="Artifact path in MLflow")
    args = ap.parse_args()

    if not args.token:
        raise SystemExit("❌ Missing Grafana API token (use --token or GRAFANA_API_TOKEN).")

    base = args.url.rstrip("/")
    t_to = args.to_ms or _now_ms()
    t_from = args.from_ms or (t_to - args.hours * 3600 * 1000)

    slug = _get_slug(base, args.token, args.uid)
    panel_ids: List[int] = [int(x) for x in args.panels.split(",") if x.strip()]

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    # Attach to MLflow run (existing or a fresh one)
    if args.mlflow_run_id:
        mlflow.start_run(run_id=args.mlflow_run_id)
    else:
        mlflow.start_run(run_name=f"grafana_export_{int(time.time())}")

    saved: List[Path] = []
    try:
        for pid in panel_ids:
            print(f"📥 Rendering panel {pid} from {args.uid} ({slug})...")
            png = _render_panel_png(
                base=base,
                token=args.token,
                uid=args.uid,
                slug=slug,
                panel_id=pid,
                t_from_ms=t_from,
                t_to_ms=t_to,
                org_id=args.org_id,
                width=args.width,
                height=args.height,
                theme=args.theme,
            )
            outfile = outdir / f"{args.uid}_{slug}_panel_{pid}_{t_from}_{t_to}.png"
            outfile.write_bytes(png)
            print(f"✅ Saved {outfile}")

            # Log to MLflow
            mlflow.log_artifact(str(outfile), artifact_path=args.artifact_path)
            saved.append(outfile)

        print(f"🧾 Logged {len(saved)} panel PNGs to MLflow under '{args.artifact_path}/'.")
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
