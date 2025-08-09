#!/usr/bin/env bash
pkill -f "mlflow ui" || true
pkill -f "prometheus" || true
pkill -f "grafana" || true
echo "🛑 Stopped known services."
