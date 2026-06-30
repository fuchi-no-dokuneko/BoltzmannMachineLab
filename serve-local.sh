#!/usr/bin/env bash
set -euo pipefail
python3 -m http.server "${PORT:-8085}" --bind "${HOST:-0.0.0.0}"
