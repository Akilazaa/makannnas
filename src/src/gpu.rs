//! GPU mining backend via OpenCL — multi-device.
//!
//! Enumerates ALL OpenCL GPU devices on the system and dispatches one worker
//! thread per device. Each device gets a non-overlapping nonce sub-space
//! (offset = device_index * 2^48), so two GPUs never hash the same nonce.
//! A shared atomic stop_flag and Mutex<Option<nonce>> coordinate first-finder-
//! wins semantics across devices.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use alloy::primitives::{keccak256, B256, U256};
use eyre::{eyre, Result};
use ocl::{flags, Buffer, Context, Device, Kernel, Platform, Program, Queue};

const KERNEL_SRC: &str = include_str!("keccak_kernel.cl");
/// Default nonces/dispatch per device. 16M saturates RTX 30/40-series at
/// ~95% GPU util out-of-box. Override via `GPU_BATCH` env var if your GPU
/// is much smaller (set lower) or much bigger (set higher, e.g. 33554432).
const DEFAULT_BATCH: usize = 1 << 24; // 16,777,216 nonces/dispatch
/// Per-device nonce sub-space offset. With 2^48 ≈ 281 trillion nonces per
/// device, even at 10 GH/s a single GPU would need ~7.8 hours to exhaust its
/// sub-space — far longer than any practical mining round.
const DEVICE_NONCE_OFFSET: u64 = 1u64 << 48;

/// Per-device OpenCL state. Context, Queue, and Program are kept alive for
/// the lifetime of the GpuMiner so kernels can be rebuilt cheaply each batch.
struct DeviceCtx {
    #[allow(dead_code)]
    context: Context,
    queue: Queue,
    program: Program,
    name: String,
}

pub struct GpuMiner {
    devices: Vec<DeviceCtx>,
    batch_size: usize,
}

impl GpuMiner {
    /// Enumerate every OpenCL GPU device on the default platform and build a
    /// per-device context/queue/program. Errors out if no device is found.
    pub fn new(batch_size: Option<usize>) -> Result<Self> {
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH);

        let platform = Platform::default();
        let device_list = Device::list_all(platform)
            .map_err(|e| eyre!("failed to enumerate OpenCL devices: {e}"))?;
        if device_list.is_empty() {
            return Err(eyre!("no OpenCL device found on default platform"));
        }

        let mut devices = Vec::with_capacity(device_list.len());
        for device in device_list {
            let name = device.name().unwrap_or_else(|_| "<unknown>".into());
            let context = Context::builder()
                .platform(platform)
                .devices(device)
                .build()
                .map_err(|e| eyre!("Context build failed for {name}: {e}"))?;
            let queue = Queue::new(&context, device, None)
                .map_err(|e| eyre!("Queue build failed for {name}: {e}"))?;
            let program = Program::builder()
                .src(KERNEL_SRC)
                .devices(device)
                .build(&context)
                .map_err(|e| eyre!("Program build failed for {name}: {e}"))?;
            devices.push(DeviceCtx { context, queue, program, name });
        }

        Ok(Self { devices, batch_size })
    }

    /// Number of GPU devices that will mine in parallel.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Names of all GPU devices in enumeration order.
    pub fn device_names(&self) -> Vec<&str> {
        self.devices.iter().map(|d| d.name.as_str()).collect()
    }

    /// First device name (kept for backward compat with callers that expect a
    /// single name).
    pub fn device_name(&self) -> &str {
        self.devices
            .first()
            .map(|d| d.name.as_str())
            .unwrap_or("<none>")
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Verify the GPU kernel matches a CPU reference on one known nonce.
    /// Test runs only on the first device — kernel source is identical across
    /// devices, so passing on one means the build is correct.
    pub fn self_test(&self) -> Result<()> {
        let dev = self
            .devices
            .first()
            .ok_or_else(|| eyre!("self-test: no devices available"))?;

        // Fabricate an "always-passes" difficulty (uint256::MAX) so any hash
        // beats it — the test asserts *which* nonce the kernel reports, not
        // whether one exists.
        let challenge = B256::from(*b"abcdefghijklmnopqrstuvwxyz012345");
        let difficulty = U256::MAX;
        let nonce_base: u64 = 12345;

        let cw = split_challenge_le(&challenge);
        let dw = split_difficulty_be(difficulty);

        let found_nonce = Buffer::<u64>::builder()
            .queue(dev.queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(1)
            .copy_host_slice(&[0u64])
            .build()?;
        let found_flag = Buffer::<i32>::builder()
            .queue(dev.queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(1)
            .copy_host_slice(&[0i32])
            .build()?;

        let kernel = Kernel::builder()
            .program(&dev.program)
            .name("mine_keccak")
            .queue(dev.queue.clone())
            .global_work_size(1usize)
            .arg(cw[0]).arg(cw[1]).arg(cw[2]).arg(cw[3])
            .arg(dw[0]).arg(dw[1]).arg(dw[2]).arg(dw[3])
            .arg(nonce_base)
            .arg(&found_nonce)
            .arg(&found_flag)
            .build()?;

        unsafe { kernel.enq()?; }
        dev.queue.finish()?;

        let mut flag = [0i32];
        found_flag.read(&mut flag[..]).enq()?;
        if flag[0] == 0 {
            return Err(eyre!(
                "self-test: GPU#0 did not report any hit against MAX difficulty"
            ));
        }

        let mut got = [0u64];
        found_nonce.read(&mut got[..]).enq()?;
        if got[0] != nonce_base {
            return Err(eyre!(
                "self-test: GPU#0 reported nonce {} but expected {}",
                got[0],
                nonce_base
            ));
        }

        let cpu_hash = cpu_hash(&challenge, U256::from(nonce_base));
        if !(U256::from_be_bytes::<32>(cpu_hash.0) < difficulty) {
            return Err(eyre!("self-test sanity: CPU hash >= MAX difficulty"));
        }

        Ok(())
    }

    /// Mine across ALL devices in parallel. Each device gets its own thread
    /// and a non-overlapping nonce sub-space. The first device to find a
    /// valid nonce wins (signals stop, others abort their current loop on
    /// next iteration). Hashrate counter is shared across all devices, so
    /// the caller sees aggregated MH/s.
    pub fn mine(
        &self,
        challenge: B256,
        difficulty: U256,
        start_nonce: u64,
        stop_flag: Arc<AtomicBool>,
        attempts_counter: Arc<AtomicU64>,
    ) -> Result<Option<u64>> {
        if self.devices.is_empty() {
            return Err(eyre!("no GPU devices available"));
        }

        let solution: Mutex<Option<u64>> = Mutex::new(None);
        let cw = split_challenge_le(&challenge);
        let dw = split_difficulty_be(difficulty);
        let batch_size = self.batch_size;

        let result: Result<()> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(self.devices.len());

            for (idx, dev) in self.devices.iter().enumerate() {
                let stop = Arc::clone(&stop_flag);
                let counter = Arc::clone(&attempts_counter);
                let solution_ref = &solution;
                let dev_start = start_nonce
                    .wrapping_add((idx as u64).wrapping_mul(DEVICE_NONCE_OFFSET));

                let h = s.spawn(move || -> Result<()> {
                    mine_on_device(
                        dev, idx, cw, dw, &challenge, difficulty,
                        dev_start, batch_size, stop, counter, solution_ref,
                    )
                });
                handles.push(h);
            }

            for (idx, h) in handles.into_iter().enumerate() {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        eprintln!("\n❌ GPU#{} thread error: {e}", idx);
                    }
                    Err(panic) => {
                        eprintln!("\n❌ GPU#{} thread panicked: {:?}", idx, panic);
                    }
                }
            }

            Ok(())
        });
        result?;

        Ok(solution.into_inner().unwrap())
    }
}

/// Per-device mining loop. Builds buffers once, then loops kernel dispatches
/// until either the shared stop_flag is set or a CPU-verified valid nonce
/// is found. On a hit, writes to the shared solution slot and signals stop.
fn mine_on_device(
    dev: &DeviceCtx,
    idx: usize,
    cw: [u64; 4],
    dw: [u64; 4],
    challenge: &B256,
    difficulty: U256,
    start_nonce: u64,
    batch_size: usize,
    stop_flag: Arc<AtomicBool>,
    attempts_counter: Arc<AtomicU64>,
    solution: &Mutex<Option<u64>>,
) -> Result<()> {
    let found_nonce = Buffer::<u64>::builder()
        .queue(dev.queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(1)
        .copy_host_slice(&[0u64])
        .build()?;
    let found_flag = Buffer::<i32>::builder()
        .queue(dev.queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(1)
        .copy_host_slice(&[0i32])
        .build()?;

    let mut nonce_base: u64 = start_nonce;
    loop {
        if stop_flag.load(Ordering::Relaxed) {
            return Ok(());
        }

        // Reset flag/nonce for this dispatch.
        found_flag.write(&[0i32][..]).enq()?;
        found_nonce.write(&[0u64][..]).enq()?;

        let kernel = Kernel::builder()
            .program(&dev.program)
            .name("mine_keccak")
            .queue(dev.queue.clone())
            .global_work_size(batch_size)
            .arg(cw[0]).arg(cw[1]).arg(cw[2]).arg(cw[3])
            .arg(dw[0]).arg(dw[1]).arg(dw[2]).arg(dw[3])
            .arg(nonce_base)
            .arg(&found_nonce)
            .arg(&found_flag)
            .build()?;

        unsafe { kernel.enq()?; }
        dev.queue.finish()?;

        attempts_counter.fetch_add(batch_size as u64, Ordering::Relaxed);

        let mut flag = [0i32];
        found_flag.read(&mut flag[..]).enq()?;
        if flag[0] != 0 {
            let mut got = [0u64];
            found_nonce.read(&mut got[..]).enq()?;
            let nonce = got[0];

            // Belt-and-braces: CPU-verify before claiming the slot.
            let h = cpu_hash(challenge, U256::from(nonce));
            if U256::from_be_bytes::<32>(h.0) < difficulty {
                let mut slot = solution.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(nonce);
                }
                stop_flag.store(true, Ordering::Relaxed);
                return Ok(());
            } else {
                eprintln!(
                    "\n⚠️  GPU#{} ({}) reported nonce {} but CPU verify failed — skipping batch",
                    idx, dev.name, nonce
                );
            }
        }

        nonce_base = nonce_base.wrapping_add(batch_size as u64);

        // Yield briefly so the stop_flag check has time to land.
        std::thread::sleep(Duration::from_micros(1));
    }
}

/// Read the 32-byte challenge as 4 little-endian u64 words (matches the lane
/// layout the kernel expects).
fn split_challenge_le(challenge: &B256) -> [u64; 4] {
    let b = challenge.as_slice();
    [
        u64::from_le_bytes(b[0..8].try_into().unwrap()),
        u64::from_le_bytes(b[8..16].try_into().unwrap()),
        u64::from_le_bytes(b[16..24].try_into().unwrap()),
        u64::from_le_bytes(b[24..32].try_into().unwrap()),
    ]
}

/// Split a uint256 into 4 big-endian u64 words (index 0 = most significant).
fn split_difficulty_be(d: U256) -> [u64; 4] {
    let b = d.to_be_bytes::<32>();
    [
        u64::from_be_bytes(b[0..8].try_into().unwrap()),
        u64::from_be_bytes(b[8..16].try_into().unwrap()),
        u64::from_be_bytes(b[16..24].try_into().unwrap()),
        u64::from_be_bytes(b[24..32].try_into().unwrap()),
    ]
}

/// CPU reference for keccak256(abi.encode(bytes32 challenge, uint256 nonce)).
fn cpu_hash(challenge: &B256, nonce: U256) -> B256 {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge.as_slice());
    buf[32..].copy_from_slice(&nonce.to_be_bytes::<32>());
    keccak256(buf)
}
