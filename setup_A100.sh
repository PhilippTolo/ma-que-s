#!/usr/bin/env bash
# Run once per new pod session to install dependencies persistently.
# Packages land in /scratch/py_packages so they survive pod restarts.
#
# Usage:
#   bash setup_A100.sh
#   source /scratch/activate_env.sh   # activates PYTHONPATH for this shell

set -euo pipefail

PKG_DIR="/scratch/py_packages"
REQ="$(dirname "$0")/shared/requirements.txt"

echo "[setup] Installing packages to $PKG_DIR ..."
pip install -r "$REQ" --target="$PKG_DIR" --upgrade

cat > /scratch/activate_env.sh << 'EOF'
#!/bin/bash
export PYTHONPATH="/scratch/py_packages:$PYTHONPATH"
echo "Environment activated — py_packages on PYTHONPATH."
EOF
chmod +x /scratch/activate_env.sh

echo ""
echo "[setup] Done."
echo "  Run:  source /scratch/activate_env.sh"
echo "  Then: python safety/data/prepare_safetybench.py --output-dir /scratch/safety_data"
