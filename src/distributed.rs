use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::error::{BackendError, Result};
use crate::inference::LlamaInferenceSession;
use crate::tensor::{CpuTensor, RuntimeDType, TensorShape};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DistributedHeader {
    pub magic: u32,
    pub is_prefill: u32, // 0 = decode, 1 = prefill
    pub seq_len: u32,
    pub position: u32,
}

impl DistributedHeader {
    pub const MAGIC: u32 = 0xCA9E111D;

    pub fn to_bytes(self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.is_prefill.to_le_bytes());
        buf[8..12].copy_from_slice(&self.seq_len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.position.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: [u8; 16]) -> Self {
        Self {
            magic: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            is_prefill: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            seq_len: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            position: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
        }
    }
}

pub fn serialize_tensor<W: Write>(writer: &mut W, tensor: &CpuTensor) -> std::io::Result<()> {
    let dims_len = tensor.shape.dims.len() as u32;
    writer.write_all(&dims_len.to_le_bytes())?;
    for &dim in &tensor.shape.dims {
        let dim_val = dim as u32;
        writer.write_all(&dim_val.to_le_bytes())?;
    }
    let data_len = tensor.data.len() as u32;
    writer.write_all(&data_len.to_le_bytes())?;

    // Write data as raw bytes (Apple Silicon to Apple Silicon is safe)
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            tensor.data.as_ptr() as *const u8,
            tensor.data.len() * std::mem::size_of::<f32>(),
        )
    };
    writer.write_all(byte_slice)?;
    Ok(())
}

pub fn deserialize_tensor<R: Read>(reader: &mut R, name: String) -> std::io::Result<CpuTensor> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let dims_len = u32::from_le_bytes(buf) as usize;
    let mut dims = Vec::with_capacity(dims_len);
    for _ in 0..dims_len {
        reader.read_exact(&mut buf)?;
        dims.push(u32::from_le_bytes(buf) as usize);
    }
    reader.read_exact(&mut buf)?;
    let data_len = u32::from_le_bytes(buf) as usize;
    let mut data = vec![0.0f32; data_len];

    // Read data as raw bytes
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            data_len * std::mem::size_of::<f32>(),
        )
    };
    reader.read_exact(byte_slice)?;

    Ok(CpuTensor {
        name,
        shape: TensorShape { dims },
        dtype: RuntimeDType::F32,
        source_type: None,
        q8_0_blocks: None,
        q8_0_packed_rows4_4x4: None,
        q8_0_packed_rows4_4x8: None,
        q8_0_runtime_storage: None,
        q8_0_file_backing: None,
        q8_0_split_file_backing: None,
        data,
    })
}

pub struct DistributedClient {
    stream: Mutex<TcpStream>,
}

impl DistributedClient {
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        Ok(Self {
            stream: Mutex::new(stream),
        })
    }

    pub fn forward_to_worker(
        &self,
        hidden: &CpuTensor,
        is_prefill: bool,
        seq_len: usize,
        position: usize,
    ) -> Result<CpuTensor> {
        let mut stream = self.stream.lock().map_err(|_| {
            BackendError::RuntimeShapeMismatch("Failed to lock TCP stream mutex".to_string())
        })?;

        let header = DistributedHeader {
            magic: DistributedHeader::MAGIC,
            is_prefill: if is_prefill { 1 } else { 0 },
            seq_len: seq_len as u32,
            position: position as u32,
        };

        // Send header
        stream
            .write_all(&header.to_bytes())
            .map_err(|source| BackendError::Io {
                path: PathBuf::from("distributed_tcp_client_write"),
                source,
            })?;

        // Send tensor
        serialize_tensor(&mut *stream, hidden).map_err(|source| BackendError::Io {
            path: PathBuf::from("distributed_tcp_client_write_tensor"),
            source,
        })?;

        stream.flush().map_err(|source| BackendError::Io {
            path: PathBuf::from("distributed_tcp_client_flush"),
            source,
        })?;

        // Read response tensor
        let response = deserialize_tensor(&mut *stream, "worker_response_tensor".to_string())
            .map_err(|source| BackendError::Io {
                path: PathBuf::from("distributed_tcp_client_read_response"),
                source,
            })?;

        Ok(response)
    }
}

pub static DISTRIBUTED_CLIENT: OnceLock<DistributedClient> = OnceLock::new();
pub static DISTRIBUTED_RANGE: OnceLock<(usize, usize)> = OnceLock::new();

pub fn run_worker_loop(addr: &str, mut session: LlamaInferenceSession) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr)?;
    tracing::info!(addr = %addr, "Distributed Worker TCP server listening");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "Worker failed to accept connection");
                continue;
            }
        };

        let _ = stream.set_nodelay(true);
        tracing::info!("Worker accepted connection from coordinator");

        loop {
            let mut header_buf = [0u8; 16];
            if let Err(e) = stream.read_exact(&mut header_buf) {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    tracing::info!("Coordinator closed connection");
                } else {
                    tracing::error!(error = %e, "Error reading header from stream");
                }
                break;
            }

            let header = DistributedHeader::from_bytes(header_buf);
            if header.magic != DistributedHeader::MAGIC {
                tracing::error!(magic = ?header.magic, "Received invalid magic header");
                break;
            }

            let input_tensor =
                match deserialize_tensor(&mut stream, "coordinator_tensor".to_string()) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!(error = %e, "Failed to deserialize input tensor");
                        break;
                    }
                };

            let is_prefill = header.is_prefill == 1;

            let output_tensor = match session.forward_worker_layers(
                input_tensor,
                is_prefill,
                header.seq_len as usize,
                header.position as usize,
            ) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to run worker forward layers");
                    break;
                }
            };

            if let Err(e) = serialize_tensor(&mut stream, &output_tensor) {
                tracing::error!(error = %e, "Failed to serialize response tensor");
                break;
            }
            let _ = stream.flush();
        }
    }
    Ok(())
}

pub fn run_network_benchmark_worker(addr: &str) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr)?;
    tracing::info!(addr = %addr, "Network benchmark worker TCP server listening");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "Worker failed to accept benchmark connection");
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        tracing::info!("Worker accepted benchmark connection from coordinator");

        loop {
            let mut header = [0u8; 16]; // [magic (4B), test_type (4B), count (4B), size (4B)]
            if let Err(e) = stream.read_exact(&mut header) {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    tracing::info!("Coordinator closed benchmark connection");
                } else {
                    tracing::error!(error = %e, "Error reading benchmark header");
                }
                break;
            }

            let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
            let test_type = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
            let count = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
            let size =
                u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;

            if magic != DistributedHeader::MAGIC {
                tracing::error!(magic = ?magic, "Received invalid magic benchmark header");
                break;
            }

            match test_type {
                0 => {
                    tracing::info!("Received termination command. Ending benchmark session.");
                    break;
                }
                1 => {
                    // Latency Test
                    tracing::info!(
                        count = count,
                        size = size,
                        "Starting Latency Test loop as receiver"
                    );
                    let mut buf = vec![0u8; size];
                    for _ in 0..count {
                        stream.read_exact(&mut buf)?;
                        stream.write_all(&buf)?;
                        stream.flush()?;
                    }
                    tracing::info!("Latency Test loop completed");
                }
                2 => {
                    // Bandwidth Test
                    let total_bytes = count * 1024 * 1024; // count is in MB
                    tracing::info!(
                        total_mb = count,
                        chunk_size = size,
                        "Starting Bandwidth Test loop as receiver"
                    );
                    let mut buf = vec![0u8; size];
                    let mut bytes_received = 0;
                    while bytes_received < total_bytes {
                        let to_read = std::cmp::min(size, total_bytes - bytes_received);
                        stream.read_exact(&mut buf[..to_read])?;
                        bytes_received += to_read;
                    }
                    // Send 1-byte ACK
                    stream.write_all(&[1u8])?;
                    stream.flush()?;
                    tracing::info!(
                        bytes_received = bytes_received,
                        "Bandwidth Test loop completed"
                    );
                }
                _ => {
                    tracing::error!(test_type = test_type, "Received unknown test type");
                    break;
                }
            }
        }
    }
    Ok(())
}

pub fn run_network_benchmark_coordinator(
    addr: &str,
    ping_count: usize,
    payload_size: usize,
    bandwidth_mb: usize,
) -> anyhow::Result<()> {
    tracing::info!(addr = %addr, "Connecting to benchmark worker...");
    let mut stream = TcpStream::connect(addr)?;
    stream.set_nodelay(true)?;
    tracing::info!("Connected to benchmark worker successfully!");

    // --- Latency Test ---
    println!("\n=== Starting TCP Latency Test ===");
    println!("Payload size: {} bytes", payload_size);
    println!("Ping count: {}", ping_count);

    let header = [
        &DistributedHeader::MAGIC.to_le_bytes()[..],
        &1u32.to_le_bytes()[..], // test_type = 1
        &(ping_count as u32).to_le_bytes()[..],
        &(payload_size as u32).to_le_bytes()[..],
    ]
    .concat();

    stream.write_all(&header)?;
    stream.flush()?;

    let payload = vec![0u8; payload_size];
    let mut response = vec![0u8; payload_size];
    let mut durations_us = Vec::with_capacity(ping_count);

    for _ in 0..ping_count {
        let started = std::time::Instant::now();
        stream.write_all(&payload)?;
        stream.flush()?;
        stream.read_exact(&mut response)?;
        let elapsed = started.elapsed().as_secs_f64() * 1_000_000.0; // in microseconds
        durations_us.push(elapsed);
    }

    let min_us = durations_us.iter().copied().fold(f64::INFINITY, f64::min);
    let max_us = durations_us.iter().copied().fold(0.0, f64::max);
    let avg_us = durations_us.iter().copied().sum::<f64>() / ping_count as f64;

    println!("--- Latency Results ---");
    println!("Round-Trip Time (RTT):");
    println!("  Min RTT: {:.2} μs", min_us);
    println!("  Avg RTT: {:.2} μs", avg_us);
    println!("  Max RTT: {:.2} μs", max_us);
    println!("One-Way Latency (RTT / 2):");
    println!("  Min Latency: {:.2} μs", min_us / 2.0);
    println!("  Avg Latency: {:.2} μs", avg_us / 2.0);
    println!("  Max Latency: {:.2} μs", max_us / 2.0);

    // --- Bandwidth Test ---
    println!("\n=== Starting TCP Bandwidth Test ===");
    let total_mb = bandwidth_mb;
    let chunk_size = 65536; // 64KB chunks
    println!("Total data to send: {} MB", total_mb);
    println!("Chunk size: {} bytes", chunk_size);

    let header = [
        &DistributedHeader::MAGIC.to_le_bytes()[..],
        &2u32.to_le_bytes()[..], // test_type = 2
        &(total_mb as u32).to_le_bytes()[..],
        &(chunk_size as u32).to_le_bytes()[..],
    ]
    .concat();

    stream.write_all(&header)?;
    stream.flush()?;

    let bw_payload = vec![0u8; chunk_size];
    let total_bytes = total_mb * 1024 * 1024;
    let mut bytes_sent = 0;

    let started = std::time::Instant::now();
    while bytes_sent < total_bytes {
        let to_send = std::cmp::min(chunk_size, total_bytes - bytes_sent);
        stream.write_all(&bw_payload[..to_send])?;
        bytes_sent += to_send;
    }
    stream.flush()?;

    // Await ACK
    let mut ack = [0u8; 1];
    stream.read_exact(&mut ack)?;
    let duration = started.elapsed();
    let duration_secs = duration.as_secs_f64();

    let mb_sent = bytes_sent as f64 / (1024.0 * 1024.0);
    let bandwidth_mb_s = mb_sent / duration_secs;
    let bandwidth_gbps = (bytes_sent as f64 * 8.0) / (duration_secs * 1_000_000_000.0);

    println!("--- Bandwidth Results ---");
    println!(
        "  Total Transferred: {:.2} MB in {:.4} seconds",
        mb_sent, duration_secs
    );
    println!(
        "  Throughput: {:.2} MB/s ({:.2} Gbps)",
        bandwidth_mb_s, bandwidth_gbps
    );

    // --- Terminate Session ---
    let header = [
        &DistributedHeader::MAGIC.to_le_bytes()[..],
        &0u32.to_le_bytes()[..], // test_type = 0 (Terminate)
        &0u32.to_le_bytes()[..],
        &0u32.to_le_bytes()[..],
    ]
    .concat();
    let _ = stream.write_all(&header);
    let _ = stream.flush();

    println!("\n=== Benchmark Completed ===");
    Ok(())
}
