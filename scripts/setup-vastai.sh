#!/bin/bash
# ============================================================================
# HASH256 Multi-GPU Miner — Auto-Setup untuk Vast.ai (Ubuntu CUDA image)
# ============================================================================
# Cara pakai:
#   1. SSH ke Vast.ai instance
#   2. cd ke folder repo (clone dari github dulu)
#   3. bash scripts/setup-vastai.sh
#   4. nano .env  (isi PRIVATE_KEY)
#   5. tmux new -s miner
#   6. set -a; source .env; set +a
#   7. ./target/release/hash-miner-rs
# ============================================================================

set -euo pipefail

echo ""
echo "=========================================="
echo "  HASH256 Multi-GPU Miner — Setup Vast.ai "
echo "=========================================="
echo ""

# --- Step 1: System dependencies ---------------------------------------------
echo "[1/5] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    build-essential pkg-config libssl-dev \
    ocl-icd-opencl-dev clinfo \
    git curl tmux nano \
    >/dev/null
echo "      OK"

# --- Step 2: NVIDIA OpenCL ICD ----------------------------------------------
echo ""
echo "[2/5] Configuring NVIDIA OpenCL ICD..."
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
echo "      OK"

# --- Step 3: Verify GPU + OpenCL --------------------------------------------
echo ""
echo "[3/5] Verifying GPU + OpenCL..."
echo ""
echo "  --- nvidia-smi ---"
if ! nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; then
    echo "  ERROR: nvidia-smi failed. Is this a GPU instance?"
    exit 1
fi

echo ""
echo "  --- clinfo (OpenCL devices) ---"
if ! clinfo -l; then
    echo "  ERROR: clinfo failed."
    exit 1
fi

DEVICE_COUNT=$(clinfo -l 2>/dev/null | grep -c "Device #" || true)
if [ "$DEVICE_COUNT" -eq 0 ]; then
    echo "  ERROR: No OpenCL GPU device detected. Check NVIDIA driver / ICD config."
    exit 1
fi
echo ""
echo "  ✅ Detected $DEVICE_COUNT OpenCL GPU device(s)"

# --- Step 4: Rust toolchain --------------------------------------------------
echo ""
echo "[4/5] Installing Rust toolchain..."
if command -v cargo >/dev/null 2>&1; then
    echo "      Rust already installed: $(rustc --version)"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable >/dev/null
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
    echo "      Installed: $(rustc --version)"
fi

# Make sure cargo is in PATH for rest of script
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

# --- Step 5: Build miner -----------------------------------------------------
echo ""
echo "[5/5] Building miner (release profile, ~5-7 min first time)..."
cargo build --release
echo "      OK"

# --- .env setup --------------------------------------------------------------
echo ""
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 .env created from template."
fi

# --- Done --------------------------------------------------------------------
cat <<'EOF'

============================================================
  ✅ Setup selesai!
============================================================

Langkah selanjutnya:

  1. Edit .env, isi PRIVATE_KEY wallet mining kamu:
        nano .env

  2. Verifikasi state kontrak (cek genesis & difficulty):
        # akan di-cek otomatis saat miner pertama kali run

  3. Run miner di tmux session (biar tetap jalan setelah disconnect):
        tmux new -s miner
        set -a; source .env; set +a
        ./target/release/hash-miner-rs

  4. Detach tmux: Ctrl+B lalu D
     Attach kembali: tmux attach -t miner

  5. Monitor GPU dari tmux session lain:
        tmux new -s monitor
        watch -n 2 nvidia-smi

============================================================
EOF
