#!/usr/bin/env bash
# Run once per new pod session to install dependencies persistently.
# Packages land in /scratch/py_packages so they survive pod restarts.
#
# Usage:
#   bash setup_A100.sh
#   source /scratch/activate_env.sh   # activates PYTHONPATH for this shell

set -euo pipefail

PKG_DIR="/scratch/py_packages"

echo "[setup] Installing packages to $PKG_DIR ..."
# trl and peft depend on torch/transformers already in the image;
# --no-deps avoids pip trying to re-download them to the custom target.
pip install trl peft --no-deps --target="$PKG_DIR" --upgrade
# wandb has no torch dependency so install normally.
pip install wandb --target="$PKG_DIR" --upgrade

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
