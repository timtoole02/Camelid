//! Qwen3-4B EAGLE-3 learned head on CUDA. BF16 checkpoint weights stay resident;
//! activations and the private KV cache use FP32. Target verification, never the
//! draft logits, decides which tokens can leave the server.

use crate::{
    eagle3::{Eagle3DraftModel, Eagle3Geometry},
    error::{BackendError, Result},
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::CompileOptions;
use std::sync::Arc;

const SOURCE: &str = r#"
extern "C" __global__ void gemv(const unsigned short* w, const float* x, float* y, unsigned cols) {
    __shared__ float s[256];
    unsigned t=threadIdx.x, row=blockIdx.x;
    float v=0.f;
    for(unsigned i=t;i<cols;i+=256) v += __uint_as_float(((unsigned)w[row*cols+i])<<16)*x[i];
    s[t]=v; __syncthreads();
    for(unsigned d=128;d;d>>=1) { if(t<d) s[t]+=s[t+d]; __syncthreads(); }
    if(t==0) y[row]=s[0];
}
extern "C" __global__ void eagle_norm(const float* x,const float* w,float* y,unsigned n,unsigned offset,float eps) {
    __shared__ float s[256]; unsigned t=threadIdx.x;
    float v=0.f; for(unsigned i=t;i<n;i+=256) v+=x[i]*x[i];
    s[t]=v; __syncthreads();
    for(unsigned d=128;d;d>>=1) { if(t<d) s[t]+=s[t+d]; __syncthreads(); }
    float scale=rsqrtf(s[0]/n+eps);
    for(unsigned i=t;i<n;i+=256) y[offset+i]=x[i]*scale*w[i];
}
extern "C" __global__ void binary(const float* a,const float* b,float* y,unsigned n,unsigned silu) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=silu ? (a[i]/(1.f+expf(-a[i])))*b[i] : a[i]+b[i];
}
extern "C" __global__ void rope(float* x,unsigned heads,unsigned pos,float theta) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=heads*64) return;
    unsigned h=i/64,d=i%64;
    float angle=pos*powf(theta,-(float)d/64.f),c=cosf(angle),s=sinf(angle);
    float a=x[h*128+d],b=x[h*128+d+64];
    x[h*128+d]=a*c-b*s; x[h*128+d+64]=b*c+a*s;
}
extern "C" __global__ void scores(const float* q,const float* keys,float* scores,unsigned cap) {
    __shared__ float s[128];
    unsigned d=threadIdx.x,h=blockIdx.y,p=blockIdx.x;
    s[d]=q[h*128+d]*keys[(p*8+h/4)*128+d]; __syncthreads();
    for(unsigned n=64;n;n>>=1) { if(d<n) s[d]+=s[d+n]; __syncthreads(); }
    if(d==0) scores[h*cap+p]=s[0]*0.08838834764831845f;
}
extern "C" __global__ void softmax(float* scores,unsigned count,unsigned cap) {
    __shared__ float s[256]; unsigned t=threadIdx.x,h=blockIdx.x;
    float v=-3.402823466e38f;
    for(unsigned p=t;p<count;p+=256) v=fmaxf(v,scores[h*cap+p]);
    s[t]=v; __syncthreads();
    for(unsigned n=128;n;n>>=1) { if(t<n) s[t]=fmaxf(s[t],s[t+n]); __syncthreads(); }
    float m=s[0]; __syncthreads(); v=0.f;
    for(unsigned p=t;p<count;p+=256) { float z=expf(scores[h*cap+p]-m); scores[h*cap+p]=z; v+=z; }
    s[t]=v; __syncthreads();
    for(unsigned n=128;n;n>>=1) { if(t<n) s[t]+=s[t+n]; __syncthreads(); }
    float inv=1.f/s[0];
    for(unsigned p=t;p<count;p+=256) scores[h*cap+p]*=inv;
}
extern "C" __global__ void context(const float* scores,const float* values,float* out,unsigned count,unsigned cap) {
    unsigned h=blockIdx.x,d=threadIdx.x; float sum=0.f;
    for(unsigned p=0;p<count;p++) sum+=scores[h*cap+p]*values[(p*8+h/4)*128+d];
    out[h*128+d]=sum;
}
extern "C" __global__ void argmax(const float* logits,unsigned* out,unsigned n) {
    __shared__ float vals[256]; __shared__ unsigned ids[256]; unsigned t=threadIdx.x;
    float v=-3.402823466e38f; unsigned id=0xffffffff;
    for(unsigned i=t;i<n;i+=256) if(logits[i]>v || (logits[i]==v && i<id)) { v=logits[i];id=i; }
    vals[t]=v;ids[t]=id; __syncthreads();
    for(unsigned d=128;d;d>>=1) {
        if(t<d && (vals[t+d]>vals[t] || (vals[t+d]==vals[t] && ids[t+d]<ids[t]))) { vals[t]=vals[t+d];ids[t]=ids[t+d]; }
        __syncthreads();
    }
    if(t==0) *out=ids[0];
}
"#;

fn error(message: impl std::fmt::Display) -> BackendError {
    BackendError::RuntimeShapeMismatch(format!("EAGLE-3 CUDA: {message}"))
}

fn launch(rows: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

struct Kernels {
    gemv: CudaFunction,
    norm: CudaFunction,
    binary: CudaFunction,
    rope: CudaFunction,
    scores: CudaFunction,
    softmax: CudaFunction,
    context: CudaFunction,
    argmax: CudaFunction,
}

impl Kernels {
    fn matvec(
        &self,
        stream: &Arc<CudaStream>,
        w: &CudaSlice<u16>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        unsafe {
            stream
                .launch_builder(&self.gemv)
                .arg(w)
                .arg(x)
                .arg(y)
                .arg(&(cols as u32))
                .launch(launch(rows, 256))
        }
        .map_err(error)?;
        Ok(())
    }
    fn norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        offset: u32,
    ) -> Result<()> {
        unsafe {
            stream
                .launch_builder(&self.norm)
                .arg(x)
                .arg(w)
                .arg(y)
                .arg(&2560u32)
                .arg(&offset)
                .arg(&1e-6f32)
                .launch(launch(1, 256))
        }
        .map_err(error)?;
        Ok(())
    }
    fn binary(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        n: usize,
        silu: bool,
    ) -> Result<()> {
        unsafe {
            stream
                .launch_builder(&self.binary)
                .arg(a)
                .arg(b)
                .arg(y)
                .arg(&(n as u32))
                .arg(&(silu as u32))
                .launch(launch(n.div_ceil(256), 256))
        }
        .map_err(error)?;
        Ok(())
    }
}

/// A complete learned Qwen head and its private authoritative cache. This first
/// CUDA scheduler drafts one token from each stable state; rejected speculative
/// target rows are never appended here.
pub struct CudaEagle3Head {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    matrices: Vec<CudaSlice<u16>>,
    norms: Vec<CudaSlice<f32>>,
    features: CudaSlice<f32>,
    embedding: CudaSlice<f32>,
    g: CudaSlice<f32>,
    combined: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    cache_k: CudaSlice<f32>,
    cache_v: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    context: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    residual: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    activated: CudaSlice<f32>,
    down: CudaSlice<f32>,
    raw: CudaSlice<f32>,
    output: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    selected: CudaSlice<u32>,
    d2t: Vec<i32>,
    theta: f32,
    capacity: usize,
    filled: usize,
    next_token: Option<u32>,
}

impl CudaEagle3Head {
    pub fn new(model: &Eagle3DraftModel, capacity: usize) -> Result<Self> {
        if model.config.geometry() != Eagle3Geometry::QWEN
            || model.config.sliding_window.is_some()
            || !(1..=4096 + crate::inference::spec_tree::TREE_MAX_NODES).contains(&capacity)
        {
            return Err(error(
                "requires the Qwen3-4B full-attention head and a bounded serving cache",
            ));
        }
        let ctx =
            std::panic::catch_unwind(|| CudaContext::new(crate::cuda::selected_device_ordinal()))
                .map_err(|_| error("CUDA driver unavailable"))?
                .map_err(error)?;
        let ptx = std::panic::catch_unwind(|| {
            cudarc::nvrtc::compile_ptx_with_opts(
                SOURCE,
                CompileOptions {
                    fmad: Some(false),
                    arch: Some("compute_61"),
                    ..Default::default()
                },
            )
        })
        .map_err(|_| error("NVRTC unavailable"))?
        .map_err(error)?;
        let module = ctx.load_module(ptx).map_err(error)?;
        let f = |name| module.load_function(name).map_err(error);
        let kernels = Kernels {
            gemv: f("gemv")?,
            norm: f("eagle_norm")?,
            binary: f("binary")?,
            rope: f("rope")?,
            scores: f("scores")?,
            softmax: f("softmax")?,
            context: f("context")?,
            argmax: f("argmax")?,
        };
        let stream = ctx.new_stream().map_err(error)?;
        let m = &model.matrices;
        let mut matrices = Vec::new();
        for matrix in [
            &m.feature_fusion,
            &m.attention_q,
            &m.attention_k,
            &m.attention_v,
            &m.attention_o,
            &m.mlp_gate,
            &m.mlp_up,
            &m.mlp_down,
            &m.lm_head,
        ] {
            let words: Vec<u16> = matrix
                .bytes
                .chunks_exact(2)
                .map(|v| u16::from_le_bytes([v[0], v[1]]))
                .collect();
            matrices.push(stream.clone_htod(&words).map_err(error)?);
        }
        let n = &model.norms;
        let norms = [&n.input, &n.hidden, &n.post_attention, &n.output]
            .into_iter()
            .map(|v| stream.clone_htod(v))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(error)?;
        let alloc = |n| stream.alloc_zeros::<f32>(n).map_err(error);
        Ok(Self {
            kernels,
            matrices,
            norms,
            features: alloc(7680)?,
            embedding: alloc(2560)?,
            g: alloc(2560)?,
            combined: alloc(5120)?,
            q: alloc(4096)?,
            k: alloc(1024)?,
            v: alloc(1024)?,
            cache_k: alloc(capacity * 1024)?,
            cache_v: alloc(capacity * 1024)?,
            scores: alloc(capacity * 32)?,
            context: alloc(4096)?,
            attn: alloc(2560)?,
            residual: alloc(2560)?,
            normed: alloc(2560)?,
            gate: alloc(9728)?,
            up: alloc(9728)?,
            activated: alloc(9728)?,
            down: alloc(2560)?,
            raw: alloc(2560)?,
            output: alloc(2560)?,
            logits: alloc(32000)?,
            selected: stream.alloc_zeros(1).map_err(error)?,
            stream,
            d2t: model.d2t_offsets.clone(),
            theta: model.config.rope_theta,
            capacity,
            filled: 0,
            next_token: None,
        })
    }
    pub fn filled(&self) -> usize {
        self.filled
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn reset(&mut self) {
        self.filled = 0;
        self.next_token = None;
    }
    pub fn next_token(&self) -> Result<u32> {
        self.next_token
            .ok_or_else(|| error("head has no stable prediction"))
    }
    /// Diagnostic readback for independent full-checkpoint numerical validation.
    pub fn logits(&self) -> Result<Vec<f32>> {
        self.stream.clone_dtoh(&self.logits).map_err(error)
    }

    /// Pair the *next* token embedding with features from the current target
    /// row. Intermediate authoritative rows need only K/V; the last row also
    /// runs the learned attention/MLP/output head and refreshes the draft token.
    pub fn append(&mut self, features: &[f32], embedding: &[f32], predict: bool) -> Result<()> {
        if features.len() != 7680
            || embedding.len() != 2560
            || self.filled >= self.capacity
            || features.iter().chain(embedding).any(|x| !x.is_finite())
        {
            return Err(error(
                "invalid feature/embedding row or exhausted head cache",
            ));
        }
        self.next_token = None;
        let s = &self.stream;
        let k = &self.kernels;
        let m = &self.matrices;
        s.memcpy_htod(features, &mut self.features).map_err(error)?;
        s.memcpy_htod(embedding, &mut self.embedding)
            .map_err(error)?;
        k.matvec(s, &m[0], &self.features, &mut self.g, 2560, 7680)?;
        k.norm(s, &self.embedding, &self.norms[0], &mut self.combined, 0)?;
        k.norm(s, &self.g, &self.norms[1], &mut self.combined, 2560)?;
        k.matvec(s, &m[2], &self.combined, &mut self.k, 1024, 5120)?;
        k.matvec(s, &m[3], &self.combined, &mut self.v, 1024, 5120)?;
        unsafe {
            s.launch_builder(&k.rope)
                .arg(&mut self.k)
                .arg(&8u32)
                .arg(&(self.filled as u32))
                .arg(&self.theta)
                .launch(launch(2, 256))
        }
        .map_err(error)?;
        s.memcpy_dtod(
            &self.k,
            &mut self
                .cache_k
                .slice_mut(self.filled * 1024..(self.filled + 1) * 1024),
        )
        .map_err(error)?;
        s.memcpy_dtod(
            &self.v,
            &mut self
                .cache_v
                .slice_mut(self.filled * 1024..(self.filled + 1) * 1024),
        )
        .map_err(error)?;
        if predict {
            k.matvec(s, &m[1], &self.combined, &mut self.q, 4096, 5120)?;
            unsafe {
                s.launch_builder(&k.rope)
                    .arg(&mut self.q)
                    .arg(&32u32)
                    .arg(&(self.filled as u32))
                    .arg(&self.theta)
                    .launch(launch(8, 256))
            }
            .map_err(error)?;
            let count = (self.filled + 1) as u32;
            let cap = self.capacity as u32;
            unsafe {
                s.launch_builder(&k.scores)
                    .arg(&self.q)
                    .arg(&self.cache_k)
                    .arg(&mut self.scores)
                    .arg(&cap)
                    .launch(LaunchConfig {
                        grid_dim: (count, 32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(error)?;
            unsafe {
                s.launch_builder(&k.softmax)
                    .arg(&mut self.scores)
                    .arg(&count)
                    .arg(&cap)
                    .launch(launch(32, 256))
            }
            .map_err(error)?;
            unsafe {
                s.launch_builder(&k.context)
                    .arg(&self.scores)
                    .arg(&self.cache_v)
                    .arg(&mut self.context)
                    .arg(&count)
                    .arg(&cap)
                    .launch(launch(32, 128))
            }
            .map_err(error)?;
            k.matvec(s, &m[4], &self.context, &mut self.attn, 2560, 4096)?;
            k.binary(s, &self.g, &self.attn, &mut self.residual, 2560, false)?;
            k.norm(s, &self.residual, &self.norms[2], &mut self.normed, 0)?;
            k.matvec(s, &m[5], &self.normed, &mut self.gate, 9728, 2560)?;
            k.matvec(s, &m[6], &self.normed, &mut self.up, 9728, 2560)?;
            k.binary(s, &self.gate, &self.up, &mut self.activated, 9728, true)?;
            k.matvec(s, &m[7], &self.activated, &mut self.down, 2560, 9728)?;
            k.binary(s, &self.residual, &self.down, &mut self.raw, 2560, false)?;
            k.norm(s, &self.raw, &self.norms[3], &mut self.output, 0)?;
            k.matvec(s, &m[8], &self.output, &mut self.logits, 32000, 2560)?;
            unsafe {
                s.launch_builder(&k.argmax)
                    .arg(&self.logits)
                    .arg(&mut self.selected)
                    .arg(&32000u32)
                    .launch(launch(1, 256))
            }
            .map_err(error)?;
            let id = s.clone_dtoh(&self.selected).map_err(error)?[0] as usize;
            let offset = *self
                .d2t
                .get(id)
                .ok_or_else(|| error("nonfinite draft logits"))?;
            let token = id as i64 + i64::from(offset);
            if !(0..151936).contains(&token) {
                return Err(error("draft-to-target mapping is out of range"));
            }
            self.next_token = Some(token as u32);
        } else {
            s.synchronize().map_err(error)?;
        }
        self.filled += 1;
        Ok(())
    }
}
