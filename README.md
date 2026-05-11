# HASH Token Multi-GPU Miner (Rust)

Native miner untuk HASH256 token (`0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc`) di Ethereum mainnet. Mendukung **multi-GPU OpenCL** dispatch — 1 process, 1 wallet, semua GPU di sistem dipakai paralel.

## Fitur Utama

- **Multi-GPU otomatis** — enumerate semua OpenCL GPU device, dispatch 1 thread per GPU dengan nonce sub-space non-overlapping (offset 2^48 per device).
- **CPU fallback** — kalau `GPU=1` ngga di-set atau OpenCL ngga tersedia, otomatis pakai semua CPU core via `std::thread::scope`.
- **Watchdog epoch** — poll block number tiap 15 detik; saat epoch berubah, semua worker di-signal stop dan round di-restart dengan challenge baru.
- **EIP-1559 gas tuning** — `PRIORITY_GWEI` + `MAX_FEE_GWEI` env, plus optional `GAS_LIMIT_OVERRIDE`.
- **Self-test kernel** — di startup, kernel OpenCL di-verify match dengan CPU reference, mencegah bug kernel mencapai produksi.
- **Single binary statis** — no Node, no Python. ~12 MB compiled.

## Expected Performance

| Hardware | Hashrate (per GPU) | Notes |
|---|---|---|
| RTX 3060 | ~700 MH/s | |
| RTX 3070 | ~1.4 GH/s | |
| RTX 3090 | ~2.0 GH/s | |
| RTX 4070 | ~1.8 GH/s | |
| **RTX 4080** | **~2.7 GH/s** | tested |
| RTX 4090 | ~5.0 GH/s | |
| RTX 5090 | ~7.5 GH/s | |
| CPU (16 core) | ~20 MH/s | fallback |

Multi-GPU = sum of all devices. 2× RTX 4080 ≈ 5.4 GH/s.

## Cara Kerja Mining

1. **Challenge** = `keccak256(abi.encodePacked(chainId, contract, miner, epoch))` — unik per wallet per epoch
2. **Proof** = `keccak256(abi.encodePacked(challenge, nonce)) < currentDifficulty`
3. **Reward** = `100 HASH >> (totalMints / 100_000)` — halving setiap 100k mints

---

## Quick Start: Vast.ai (Multi-GPU)

Sewa instance Vast.ai dengan template **`nvidia/cuda:12.x-devel-ubuntu22.04`** dan ≥ 1 GPU NVIDIA (rekomendasi RTX 4080/4090).

```bash
# 1. SSH ke Vast.ai
# 2. Clone repo + jalanin auto-setup script
git clone https://github.com/USERNAME/REPO_KAMU.git
cd REPO_KAMU
bash scripts/setup-vastai.sh

# 3. Edit .env, isi PRIVATE_KEY
nano .env

# 4. Run di tmux session
tmux new -s miner
set -a; source .env; set +a
./target/release/hash-miner-rs

# Detach: Ctrl+B lalu D
# Attach: tmux attach -t miner
```

Setup script otomatis:
- Install dependencies (`build-essential`, `ocl-icd-opencl-dev`, `clinfo`, `tmux`)
- Konfigurasi NVIDIA OpenCL ICD (`/etc/OpenCL/vendors/nvidia.icd`)
- Install Rust toolchain (kalau belum)
- Verifikasi GPU + OpenCL detection
- Build miner dengan profile release
- Bikin `.env` dari template

---

## Build Manual (Local Development)

### Requirements

- Rust toolchain ≥ 1.75 — install via [rustup.rs](https://rustup.rs)
- (Untuk GPU mode) OpenCL headers + libOpenCL.so:
  - **Linux**: `apt install ocl-icd-opencl-dev clinfo`
  - **Windows**: install OpenCL SDK dari vendor GPU + MSVC Build Tools
- Wallet Ethereum dengan ETH untuk gas

### Build

```bash
# Linux/Mac
cargo build --release

# Windows (PowerShell)
cargo build --release
```

Binary keluar di `target/release/hash-miner-rs` (atau `.exe` di Windows).

Default features sudah include `gpu` — kalau mau CPU-only build:
```bash
cargo build --release --no-default-features
```

---

## Environment Variables

Liat [`.env.example`](./.env.example) untuk dokumentasi lengkap. Ringkasnya:

| Var | Default | Keterangan |
|---|---|---|
| `PRIVATE_KEY` | _(prompt)_ | **WAJIB**. Wallet mining (64 hex). |
| `RPC_URL` | `https://eth.llamarpc.com` | Endpoint Ethereum mainnet RPC. |
| `GPU` | _(unset)_ | Set `1` untuk aktifin GPU OpenCL backend. |
| `GPU_BATCH` | `16777216` | Nonces/dispatch per GPU (16M default = saturasi RTX 30/40). |
| `MINER_THREADS` | _all cores_ | Banyak CPU thread (untuk CPU mode). |
| `PRIORITY_GWEI` | `5` | Tip ke validator (EIP-1559). |
| `MAX_FEE_GWEI` | `100` | Ceiling absolute fee. |
| `GAS_LIMIT_OVERRIDE` | _(auto)_ | Optional, force gas limit angka. |

Cara load `.env` saat run:

```bash
# Bash/Linux:
set -a; source .env; set +a; ./target/release/hash-miner-rs

# PowerShell:
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([^#=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
  }
}
.\target\release\hash-miner-rs.exe
```

Atau langsung: `dotenvy` otomatis load `.env` dari working directory.

---

## Tuning untuk Multi-GPU

Setelah miner jalan, monitor saturasi GPU:

```bash
# Tmux pane lain
watch -n 2 'nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv,noheader'
```

**Target**:
- GPU util ≥ **95%** di semua device
- Power draw ≈ TGP tiap GPU (RTX 4080 = 320W)

Kalau util < 90%:
```bash
# Naikin batch (per GPU)
GPU=1 GPU_BATCH=33554432 ./target/release/hash-miner-rs   # 32M
GPU=1 GPU_BATCH=67108864 ./target/release/hash-miner-rs   # 64M
```

Kalau out-of-memory:
```bash
GPU=1 GPU_BATCH=8388608 ./target/release/hash-miner-rs    # 8M
GPU=1 GPU_BATCH=4194304 ./target/release/hash-miner-rs    # 4M
```

---

## Stop & Statistik

`Ctrl+C` — signal semua worker stop setelah attempt current selesai. Print final statistics:

```
📊 Final Statistics:
   Total Attempts: 134,217,728
   Successful Mints: 1
   Average Hash Rate: 5,400,000,000.00 H/s
   Mining Duration: 53.42 seconds
```

---

## Security

- **JANGAN commit `.env`** ke git. `.gitignore` sudah handle ini, tapi double-check.
- **JANGAN pakai wallet utama** untuk mining. Bikin wallet baru, isi 0.05–0.1 ETH untuk gas.
- **VPS = trust pihak ketiga**. Operator Vast.ai punya akses fisik ke disk + RAM. Private key bisa di-dump teori­tis. Sweep saldo HASH ke wallet utama berkala.
- **RPC publik bisa di-MITM atau rate-limit**. Pakai Alchemy/Infura/MEV Blocker punya sendiri.
- **Solo mining mainnet = probabilistik**. Liat `currentDifficulty()` × hashrate-mu untuk estimasi expected solve time. Hentikan kalau gas + sewa GPU > expected reward.

---

## Troubleshooting

| Error | Penyebab | Fix |
|---|---|---|
| `linker 'cc' not found` | Build tools belum install | `apt install build-essential` |
| `no OpenCL device found` | ICD belum konek | `echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd` |
| `failed openssl-sys build` | libssl-dev belum install | `apt install libssl-dev pkg-config` |
| `GPU self-test FAILED` | Kernel/driver issue | Cek `clinfo`, update NVIDIA driver |
| `Invalid private key length` | `.env` salah format | Pastiin 64 hex chars (prefix 0x optional) |
| `Genesis is not complete yet` | Kontrak belum buka mining | Tunggu, jangan run miner (bakal revert) |
| `Transaction reverted (status=0)` | Tx kalah race / block cap penuh | Naikin `PRIORITY_GWEI` (5 → 10) |
| RPC error berulang | Free RPC rate-limited | Ganti `RPC_URL` ke Alchemy/Infura/MEVBlocker |

---

## License

MIT — risiko ditanggung sendiri. Tidak ada garansi profitabilitas.
