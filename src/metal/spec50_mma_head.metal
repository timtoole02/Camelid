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
