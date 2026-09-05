#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
uint64_t rng=0x1234567877665544ull;
uint32_t rand32(){rng ^= rng <<13; rng ^= rng>>7; rng ^= rng<<17;return rng;}
int main(int argc,char**argv){@autoreleasepool{
 if(argc<3){fprintf(stderr,"usage: head_bench shader-list rows [k]\n");return 1;}
 uint32_t rows=atoi(argv[2]), n_sb=15, k=argc>3?atoi(argv[3]):8;
 float cap=30;
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); auto queue=[dev newCommandQueue];
 auto buf=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
 auto w=buf(size_t(rows)*n_sb*210),sc=buf(k*n_sb*4),q=buf(k*n_sb*256),perm=buf(k*n_sb*256*4),out=buf(k*rows*4);
 auto wp=(uint8_t*)w.contents;
 for(size_t i=0;i<size_t(rows)*n_sb;i++){for(int j=0;j<208;j++)wp[i*210+j]=rand32(); uint16_t scale=0x1900+(rand32()&511);memcpy(wp+i*210+208,&scale,2);}
 for(uint32_t i=0;i<k*n_sb;i++)((float*)sc.contents)[i]=float((rand32()%100)+1)*0.0001f;
 for(uint32_t i=0;i<k*n_sb*256;i++)((int8_t*)q.contents)[i]=rand32();
 NSString *list=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil];
 std::vector<uint32_t> oracle;
 bool failed=false;
 for(NSString *line in [list componentsSeparatedByString:@"\n"]){if(!line.length)continue;
 auto fields=[line componentsSeparatedByString:@" "];NSString* file=fields[0];int rb=[fields[1]intValue],sg=[fields[2]intValue];
 NSError* err=nil;NSString* source=[NSString stringWithContentsOfFile:file encoding:NSUTF8StringEncoding error:&err];
 MTLCompileOptions* opts=[MTLCompileOptions new];opts.fastMathEnabled=NO;
 auto lib=[dev newLibraryWithSource:source options:opts error:&err];if(!lib){fprintf(stderr,"COMPILE %s %s\n",file.UTF8String,err.localizedDescription.UTF8String);continue;}
 auto expandfn=[lib newFunctionWithName:@"q6k_spec50_expand_f16"];
 if(!expandfn)expandfn=[lib newFunctionWithName:@"q6k_spec50_mma_expand_f16"];
 auto expand=[dev newComputePipelineStateWithFunction:expandfn error:&err];
 auto fn=[lib newFunctionWithName:[NSString stringWithFormat:@"q6k_spec50_batch_k%u",k]];
 if(!fn)fn=[lib newFunctionWithName:[NSString stringWithFormat:@"q6k_spec50_mma_k%u",k]];
 auto pipe=[dev newComputePipelineStateWithFunction:fn error:&err];
 if(!pipe){fprintf(stderr,"PIPE %s %s\n",file.UTF8String,err.localizedDescription.UTF8String);continue;}
 std::vector<double> times;
 for(int rep=0;rep<9;rep++){
 auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
 [e setComputePipelineState:expand];[e setBuffer:q offset:0 atIndex:0];[e setBuffer:perm offset:0 atIndex:1];[e setBytes:&n_sb length:4 atIndex:2];[e setBytes:&k length:4 atIndex:3];
 [e dispatchThreadgroups:MTLSizeMake((k*n_sb*256+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 [e setComputePipelineState:pipe];[e setBuffer:sc offset:0 atIndex:0];[e setBuffer:perm offset:0 atIndex:1];[e setBuffer:w offset:0 atIndex:2];[e setBuffer:out offset:0 atIndex:3];[e setBytes:&n_sb length:4 atIndex:4];[e setBytes:&rows length:4 atIndex:5];[e setBytes:&cap length:4 atIndex:6];
 [e dispatchThreadgroups:MTLSizeMake((rows+sg*rb-1)/(sg*rb),1,1) threadsPerThreadgroup:MTLSizeMake(32*sg,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
 if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"GPU ERROR %s\n",cb.error.localizedDescription.UTF8String);return 2;}
 if(rep>=2)times.push_back((cb.GPUEndTime-cb.GPUStartTime)*1000.0);
 }
 size_t bad=0;auto bits=(uint32_t*)out.contents;
 if(oracle.empty())oracle.assign(bits,bits+size_t(k)*rows);else for(size_t i=0;i<size_t(k)*rows;i++)if(bits[i]!=oracle[i]){if(bad++<3)fprintf(stderr,"bad %zu: %08x %08x\n",i,oracle[i],bits[i]);}
 if(bad)failed=true;
 std::sort(times.begin(),times.end());printf("%s rows=%u K=%u bad=%zu median_ms=%.4f min_ms=%.4f max_ms=%.4f\n",file.UTF8String,rows,k,bad,times[times.size()/2],times[0],times.back());fflush(stdout);
 }
 return failed?2:0;
}}
