// Exact Q6_K x Q8_K K8 head: integer group dots on the matrix unit, with
// the established 32-lane floating-point fold restored before simd_sum.
// Only the 3840-wide 12B head is admitted by the host (60 units per row).
// Each dot contains 16 products of integers |w|<=32, |q|<=128. Every prefix
// is exactly representable in f32; multiplying by the signed Q6_K subscale
// remains an exact integer <=2^23. Recombine the four groups in int32.
// A simdgroup computes u and u+32 in that order. Scratch transposes those
// partials back to the original unit-to-lane assignment, keeping the same
// scale expression, simd_sum, and softcap. Fast math must remain disabled.
#include <metal_stdlib>
using namespace metal;
#ifndef MMA_SG
#define MMA_SG 4
#endif
kernel void q6k_spec50_mma_expand_f16(device const char* quants [[buffer(0)]],device half* out [[buffer(1)]],constant uint& n_sb [[buffer(2)]],constant uint& k_batch [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
 const uint units=n_sb*4u,hidden=n_sb*256u;if(gid>=k_batch*hidden)return;
 const uint cell=gid&1u,lane=(gid>>1u)&31u,kt=(gid>>6u)&1u,g=(gid>>7u)&3u,u=gid>>9u;
 const uint t=4u*((lane>>3u)&1u)+2u*(lane&1u)+cell,l=kt*8u+4u*(lane>>4u)+((lane&7u)>>1u);
 const uint sb=u>>2u,quarter=u&3u,h=quarter>>1u,s=quarter&1u;
 out[gid]=half(quants[t*hidden+sb*256u+h*128u+s*16u+g*32u+l]);
}
inline uint2 head_coord(uint lane){return uint2(4u*((lane>>3u)&1u)+2u*(lane&1u),4u*(lane>>4u)+((lane&7u)>>1u));}
template<uint KB>
void mma_head(device const float* input_scales, device const half* input_perm,device const uchar* weight_blocks,device float* output,uint n_sb,uint rows,float softcap,uint tile,uint sgitg,uint lane,threadgroup float* partials){
 const uint row0=tile*8u,units=n_sb*4u;const uint2 fc=head_coord(lane);const uint row=min(row0+fc.y,rows-1u);
 for(uint unit0=sgitg;unit0<32u;unit0+=MMA_SG){
  float accum[2]={0.0f,0.0f};
  for(uint pass=0;pass<2u;pass++){
   const uint u=unit0+pass*32u;if(u>=units)continue;
   const uint sb=u>>2u,quarter=u&3u,h=quarter>>1u,s=quarter&1u;
   device const uchar* block=weight_blocks+(ulong(row)*n_sb+sb)*210ul;
   device const char* ws=reinterpret_cast<device const char*>(block+192u);
   // Load the packed Q6 bytes once. Both nibbles and all four high-bit
   // pairs are reused below; packed loads also keep each two-byte fragment
   // coalesced. This changes only decoding, before exact integer dot products.
   uchar2 ql0[2],ql1[2],qh0[2];
   #pragma unroll
   for(uint kt=0;kt<2u;kt++) {
    const uint j=s*16u+kt*8u+fc.x;
    ql0[kt]=uchar2(*reinterpret_cast<device const packed_uchar2*>(block+h*64u+j));
    ql1[kt]=uchar2(*reinterpret_cast<device const packed_uchar2*>(block+h*64u+32u+j));
    qh0[kt]=uchar2(*reinterpret_cast<device const packed_uchar2*>(block+128u+h*32u+j));
   }
   int isum[2]={0,0};
   #pragma unroll
   for(uint g=0;g<4u;g++){
    simdgroup_float8x8 dots=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    #pragma unroll
    for(uint kt=0;kt<2u;kt++){
     simdgroup_half8x8 weights,activations;
     const uint2 qlv=uint2((g&1u)?ql1[kt]:ql0[kt]),qhv=uint2(qh0[kt]);
     const uint2 low=g<2u?(qlv&15u):(qlv>>4u),high=((qhv>>(g*2u))&3u)<<4u;
     const half2 w=half2(int2(low|high)-32);
     weights.thread_elements()[0]=w.x;weights.thread_elements()[1]=w.y;
     #pragma unroll
     for(uint cell=0;cell<2u;cell++){
      const uint t=fc.x+cell;
      activations.thread_elements()[cell]=t<KB?input_perm[((u*4u+g)*2u+kt)*64u+lane*2u+cell]:half(0.0f);
     }
     simdgroup_multiply_accumulate(dots,weights,activations,dots);
    }
    const float subscale=float(ws[8u*h+s+2u*g]);
    isum[0]+=int(subscale*dots.thread_elements()[0]);
    isum[1]+=int(subscale*dots.thread_elements()[1]);
   }
   const float weight_scale=float(*reinterpret_cast<device const half*>(block+208u));
   #pragma unroll
   for(uint cell=0;cell<2u;cell++){
    const uint t=fc.x+cell;
    if(t<KB){const float in_scale=input_scales[t*n_sb+sb];accum[cell]=accum[cell]+(weight_scale*in_scale)*float(isum[cell]);}
   }
  }
  #pragma unroll
  for(uint cell=0;cell<2u;cell++)partials[unit0*64u+fc.y*8u+fc.x+cell]=accum[cell];
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint outcell=sgitg;outcell<8u*KB;outcell+=MMA_SG){
  const uint r=outcell/KB,t=outcell%KB;
  const float partial=partials[lane*64u+r*8u+t];
  float value=simd_sum(partial);
  if(lane==0u&&row0+r<rows){if(softcap>0.0f)value=tanh(value/softcap)*softcap;output[ulong(t)*rows+row0+r]=value;}
 }
}
#define ENTRY(K) \
kernel void q6k_spec50_mma_k##K(device const float* scales [[buffer(0)]],device const half* perm [[buffer(1)]],device const uchar* weights [[buffer(2)]],device float* output [[buffer(3)]],constant uint& n_sb [[buffer(4)]],constant uint& rows [[buffer(5)]],constant float& cap [[buffer(6)]],uint tile [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {threadgroup float partials[2048];mma_head<K>(scales,perm,weights,output,n_sb,rows,cap,tile,sg,lane,partials);}
ENTRY(8)

kernel void q6k_spec50_mma_expand16_f16(device const char* quants [[buffer(0)]],device half* out [[buffer(1)]],constant uint& n_sb [[buffer(2)]],constant uint& k_batch [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
 const uint units=n_sb*4u,hidden=n_sb*256u;if(gid>=k_batch*hidden)return;
 const uint group=gid/(units*512u),local=gid%(units*512u);
 const uint cell=local&1u,lane=(local>>1u)&31u,kt=(local>>6u)&1u,g=(local>>7u)&3u,u=local>>9u;
 const uint t=group*8u+4u*((lane>>3u)&1u)+2u*(lane&1u)+cell,l=kt*8u+4u*(lane>>4u)+((lane&7u)>>1u);
 const uint sb=u>>2u,quarter=u&3u,h=quarter>>1u,s=quarter&1u;
 out[gid]=half(quants[t*hidden+sb*256u+h*128u+s*16u+g*32u+l]);
}
template<uint SG>
void mma_head16(device const float* input_scales, device const half* input_perm,device const uchar* weight_blocks,device float* output,uint n_sb,uint rows,float softcap,uint tile,uint sgitg,uint lane,threadgroup float* partials){
 const uint row0=tile*8u,units=n_sb*4u;const uint2 fc=head_coord(lane);const uint row=min(row0+fc.y,rows-1u);
 for(uint unit0=sgitg;unit0<32u;unit0+=SG){
  float accum[2][2]={{0.0f,0.0f},{0.0f,0.0f}};
  for(uint pass=0;pass<2u;pass++){
   const uint u=unit0+pass*32u;if(u>=units)continue;
   const uint sb=u>>2u,quarter=u&3u,h=quarter>>1u,s=quarter&1u;
   device const uchar* block=weight_blocks+(ulong(row)*n_sb+sb)*210ul;
   device const char* ws=reinterpret_cast<device const char*>(block+192u);
   // Load the packed Q6 bytes once. Both nibbles and all four high-bit
   // pairs are reused below; packed loads also keep each two-byte fragment
   // coalesced. This changes only decoding, before exact integer dot products.
   uchar2 ql0[2],ql1[2],qh0[2];
   #pragma unroll
   for(uint kt=0;kt<2u;kt++) {
    const uint j=s*16u+kt*8u+fc.x;
    ql0[kt]=uchar2(*reinterpret_cast<device const packed_uchar2*>(block+h*64u+j));
    ql1[kt]=uchar2(*reinterpret_cast<device const packed_uchar2*>(block+h*64u+32u+j));
    qh0[kt]=uchar2(*reinterpret_cast<device const packed_uchar2*>(block+128u+h*32u+j));
   }
   int isum[2][2]={{0,0},{0,0}};
   #pragma unroll
   for(uint g=0;g<4u;g++){
    simdgroup_float8x8 dots[2];
    #pragma unroll
    for(uint bg=0;bg<2u;bg++) dots[bg]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    #pragma unroll
    for(uint kt=0;kt<2u;kt++){
     simdgroup_half8x8 weights;
     const uint2 qlv=uint2((g&1u)?ql1[kt]:ql0[kt]),qhv=uint2(qh0[kt]);
     const uint2 low=g<2u?(qlv&15u):(qlv>>4u),high=((qhv>>(g*2u))&3u)<<4u;
     const half2 w=half2(int2(low|high)-32);
     weights.thread_elements()[0]=w.x;weights.thread_elements()[1]=w.y;
     #pragma unroll
     for(uint bg=0;bg<2u;bg++) {
      simdgroup_half8x8 activations;
      #pragma unroll
      for(uint cell=0;cell<2u;cell++) {
       activations.thread_elements()[cell]=input_perm[(((bg*units+u)*4u+g)*2u+kt)*64u+lane*2u+cell];
      }
      simdgroup_multiply_accumulate(dots[bg],weights,activations,dots[bg]);
     }
    }
    const float subscale=float(ws[8u*h+s+2u*g]);
    #pragma unroll
    for(uint bg=0;bg<2u;bg++) {
     isum[bg][0]+=int(subscale*dots[bg].thread_elements()[0]);
     isum[bg][1]+=int(subscale*dots[bg].thread_elements()[1]);
    }
   }
   const float weight_scale=float(*reinterpret_cast<device const half*>(block+208u));
   #pragma unroll
   for(uint bg=0;bg<2u;bg++) {
    #pragma unroll
    for(uint cell=0;cell<2u;cell++) {
     const uint t=bg*8u+fc.x+cell;
     const float in_scale=input_scales[t*n_sb+sb];
     accum[bg][cell]=accum[bg][cell]+(weight_scale*in_scale)*float(isum[bg][cell]);
    }
   }
  }
  #pragma unroll
  for(uint bg=0;bg<2u;bg++) {
   #pragma unroll
   for(uint cell=0;cell<2u;cell++)partials[unit0*128u+fc.y*16u+bg*8u+fc.x+cell]=accum[bg][cell];
  }
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint outcell=sgitg;outcell<128u;outcell+=SG){
  const uint r=outcell/16u,t=outcell%16u;
  const float partial=partials[lane*128u+r*16u+t];
  float value=simd_sum(partial);
  if(lane==0u&&row0+r<rows){if(softcap>0.0f)value=tanh(value/softcap)*softcap;output[ulong(t)*rows+row0+r]=value;}
 }
}
// K16 owns two independent eight-column groups and shares only integer
// Q6 fragment decoding. Each group's two-pass unit fold and final simd_sum
// remain the K8 oracle. The established K8 entry above is unchanged.
#define ENTRY16(SG) \
kernel void q6k_spec50_mma_k16_sg##SG(device const float* scales [[buffer(0)]],device const half* perm [[buffer(1)]],device const uchar* weights [[buffer(2)]],device float* output [[buffer(3)]],constant uint& n_sb [[buffer(4)]],constant uint& rows [[buffer(5)]],constant float& cap [[buffer(6)]],uint tile [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {threadgroup float partials[4096];mma_head16<SG>(scales,perm,weights,output,n_sb,rows,cap,tile,sg,lane,partials);}
ENTRY16(4)
ENTRY16(8)
#undef ENTRY16

// ---------------------------------------------------------------------------
// Instruction-reduced K8 sibling ("lean"), reached only through
// CAMELID_GEMMA4_SPEC50_HEAD_FORM. Everything above this line is untouched and
// is what the unset selector still compiles.
//
// The head is issue-bound, not DRAM-bound: at K=8 one (simdgroup, unit) step
// streams 384 bytes of Q6_K and issues on the order of 280 SIMD instructions,
// of which only 8 are matrix multiplies. The three changes below remove
// instructions without touching a single arithmetic expression:
//
//   1. ((qh >> 2g) & 3) << 4 is rewritten as ((qh << shl) >> shr) & 48 with
//      compile-time shl/shr. For qh in [0,255] the two forms select the same
//      two bits and land them at positions 4 and 5, so the integer is equal
//      for every input, not merely for the ones we tested.
//   2. The activation guard `t < KB` is dropped. With KB == 8 and
//      t = fc.x + cell in {0..7} it is unconditionally true, so the selected
//      value never changes; the pair is then fetched with one 4-byte
//      packed_half2 load (the slot is 4-byte aligned: lane*2 halves).
//   3. PF == 1 hoists both unit passes' device loads above both passes'
//      arithmetic, doubling the loads a lane has in flight. The passes are
//      still consumed in ascending order into the same `accum`, so the fold is
//      unchanged; a pass past `units` is loaded from a clamped (in-bounds)
//      unit and then discarded, exactly as `continue` discarded it before.
//
// The per-cell fold order, the subscale application, the int32 recombination,
// the (weight_scale * in_scale) product, the partials transpose, simd_sum and
// the softcap are all copied verbatim from mma_head<8>.
// ---------------------------------------------------------------------------
struct spec50_lean_unit {
 uchar2 ql0[2],ql1[2],qh0[2];
 float sub[4];
 float weight_scale;
 float in_scale[2];
};

inline void spec50_lean_load(device const uchar* weight_blocks,device const float* input_scales,uint row,uint n_sb,uint u,uint2 fc,thread spec50_lean_unit& su){
 const uint sb=u>>2,quarter=u&3u,h=quarter>>1,s=quarter&1u;
 device const uchar* block=weight_blocks+(ulong(row)*n_sb+sb)*210ul;
 device const char* ws=reinterpret_cast<device const char*>(block+192u);
 #pragma unroll
 for(uint kt=0;kt<2u;kt++) {
  const uint j=s*16u+kt*8u+fc.x;
  su.ql0[kt]=uchar2(*reinterpret_cast<device const packed_uchar2*>(block+h*64u+j));
  su.ql1[kt]=uchar2(*reinterpret_cast<device const packed_uchar2*>(block+h*64u+32u+j));
  su.qh0[kt]=uchar2(*reinterpret_cast<device const packed_uchar2*>(block+128u+h*32u+j));
 }
 #pragma unroll
 for(uint g=0;g<4u;g++) su.sub[g]=float(ws[8u*h+s+2u*g]);
 su.weight_scale=float(*reinterpret_cast<device const half*>(block+208u));
 #pragma unroll
 for(uint cell=0;cell<2u;cell++) su.in_scale[cell]=input_scales[(fc.x+cell)*n_sb+sb];
}

template<uint KB>
inline void spec50_lean_accum(device const half* input_perm,uint u,uint lane,thread const spec50_lean_unit& su,thread float* accum){
 static_assert(KB==8u,"the lean SPEC50 head owns the eight-column tile only");
 device const half* perm=input_perm+u*512u+lane*2u;
 int isum[2]={0,0};
 #pragma unroll
 for(uint g=0;g<4u;g++){
  simdgroup_float8x8 dots=make_filled_simdgroup_matrix<float,8,8>(0.0f);
  #pragma unroll
  for(uint kt=0;kt<2u;kt++){
   simdgroup_half8x8 weights,activations;
   const uint2 qlv=uint2((g&1u)?su.ql1[kt]:su.ql0[kt]),qhv=uint2(su.qh0[kt]);
   const uint shl=(g<2u)?(4u-2u*g):0u,shr=(g<2u)?0u:(2u*g-4u);
   const uint2 low=g<2u?(qlv&15u):(qlv>>4u),high=((qhv<<shl)>>shr)&48u;
   const half2 w=half2(int2(low|high)-32);
   weights.thread_elements()[0]=w.x;weights.thread_elements()[1]=w.y;
   const half2 a=half2(*reinterpret_cast<device const packed_half2*>(perm+(g*2u+kt)*64u));
   activations.thread_elements()[0]=a.x;activations.thread_elements()[1]=a.y;
   simdgroup_multiply_accumulate(dots,weights,activations,dots);
  }
  isum[0]+=int(su.sub[g]*dots.thread_elements()[0]);
  isum[1]+=int(su.sub[g]*dots.thread_elements()[1]);
 }
 #pragma unroll
 for(uint cell=0;cell<2u;cell++)accum[cell]=accum[cell]+(su.weight_scale*su.in_scale[cell])*float(isum[cell]);
}

template<uint KB,uint SG,uint PF>
void mma_head_lean(device const float* input_scales, device const half* input_perm,device const uchar* weight_blocks,device float* output,uint n_sb,uint rows,float softcap,uint tile,uint sgitg,uint lane,threadgroup float* partials){
 const uint row0=tile*8u,units=n_sb*4u;const uint2 fc=head_coord(lane);const uint row=min(row0+fc.y,rows-1u);
 for(uint unit0=sgitg;unit0<32u;unit0+=SG){
  float accum[2]={0.0f,0.0f};
  if(PF==0u){
   #pragma unroll
   for(uint pass=0;pass<2u;pass++){
    const uint u=unit0+pass*32u;if(u>=units)continue;
    spec50_lean_unit su;
    spec50_lean_load(weight_blocks,input_scales,row,n_sb,u,fc,su);
    spec50_lean_accum<KB>(input_perm,u,lane,su,accum);
   }
  } else {
   spec50_lean_unit su[2];bool live[2];uint uu[2];
   #pragma unroll
   for(uint pass=0;pass<2u;pass++){
    const uint u=unit0+pass*32u;
    live[pass]=u<units;
    uu[pass]=live[pass]?u:unit0;
    spec50_lean_load(weight_blocks,input_scales,row,n_sb,uu[pass],fc,su[pass]);
   }
   #pragma unroll
   for(uint pass=0;pass<2u;pass++){
    if(!live[pass])continue;
    spec50_lean_accum<KB>(input_perm,uu[pass],lane,su[pass],accum);
   }
  }
  #pragma unroll
  for(uint cell=0;cell<2u;cell++)partials[unit0*64u+fc.y*8u+fc.x+cell]=accum[cell];
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint outcell=sgitg;outcell<8u*KB;outcell+=SG){
  const uint r=outcell/KB,t=outcell%KB;
  const float partial=partials[lane*64u+r*8u+t];
  float value=simd_sum(partial);
  if(lane==0u&&row0+r<rows){if(softcap>0.0f)value=tanh(value/softcap)*softcap;output[ulong(t)*rows+row0+r]=value;}
 }
}
#define ENTRY_LEAN(K,SGV,PFV) \
kernel void q6k_spec50_mma_lean_k##K##_sg##SGV##_pf##PFV(device const float* scales [[buffer(0)]],device const half* perm [[buffer(1)]],device const uchar* weights [[buffer(2)]],device float* output [[buffer(3)]],constant uint& n_sb [[buffer(4)]],constant uint& rows [[buffer(5)]],constant float& cap [[buffer(6)]],uint tile [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {threadgroup float partials[2048];mma_head_lean<K,SGV,PFV>(scales,perm,weights,output,n_sb,rows,cap,tile,sg,lane,partials);}
ENTRY_LEAN(8,1,0)
ENTRY_LEAN(8,2,0)
ENTRY_LEAN(8,4,0)
ENTRY_LEAN(8,8,0)
ENTRY_LEAN(8,16,0)
ENTRY_LEAN(8,1,1)
ENTRY_LEAN(8,2,1)
ENTRY_LEAN(8,4,1)
ENTRY_LEAN(8,8,1)
ENTRY_LEAN(8,16,1)
#undef ENTRY_LEAN
