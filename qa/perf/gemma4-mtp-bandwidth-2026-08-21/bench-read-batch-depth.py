#!/usr/bin/env python3
"""Does small per-layer batch size explain 1.87 GB/s vs the device's 2.60?"""
import os, time, random, threading, fcntl
operator_home = os.environ.get("CAMELID_OPERATOR_HOME") or os.environ.get("HOME")
if not operator_home:
    raise RuntimeError("set CAMELID_OPERATOR_HOME or HOME")
model_root = os.environ.get("CAMELID_MODEL_ROOT") or os.path.join(operator_home, "models")
PATH = os.environ.get("CAMELID_GEMMA4_CGHOST") or os.path.join(
    model_root, "gemma4-mtp-pair", "gemma-4-26B_q4_0-it.v3.cghost"
)
REC=3_345_408; F_NOCACHE=48
nrec=os.path.getsize(PATH)//REC
random.seed(7)

def run(batch, nthreads, batches=40):
    fds=[]
    for _ in range(nthreads):
        fd=os.open(PATH, os.O_RDONLY); fcntl.fcntl(fd,F_NOCACHE,1); fds.append(fd)
    total=0; t0=time.monotonic()
    for _ in range(batches):
        offs=[random.randrange(0,nrec-1)*REC for _ in range(batch)]
        got=[0]*nthreads
        def w(i):
            n=0
            for j in range(i,len(offs),nthreads): n+=len(os.pread(fds[i],REC,offs[j]))
            got[i]=n
        ts=[threading.Thread(target=w,args=(i,)) for i in range(nthreads)]
        [t.start() for t in ts]; [t.join() for t in ts]   # barrier per batch, like the engine
        total+=sum(got)
    dt=time.monotonic()-t0
    for fd in fds: os.close(fd)
    return total/dt/1e9

print("Engine issues ~7 reads per layer, 4 threads, with a barrier before the next layer.\n")
print(f"{'batch':>6s} {'threads':>8s} {'GB/s':>7s}   note")
for batch,note in ((7,"<- the engine's actual per-layer batch"),(14,""),(28,""),(56,""),(192,"<- my earlier benchmark")):
    bw=run(batch,4)
    print(f"{batch:6d} {4:8d} {bw:7.2f}   {note}")
print()
for nt in (6,8):
    print(f"{7:6d} {nt:8d} {run(7,nt):7.2f}   engine batch, more threads")
