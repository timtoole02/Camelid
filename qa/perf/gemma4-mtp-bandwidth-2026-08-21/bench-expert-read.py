#!/usr/bin/env python3
"""Replicate the engine's expert-record read pattern: random 3,345,408 B preads."""
import os, sys, time, random, threading, fcntl
operator_home = os.environ.get("CAMELID_OPERATOR_HOME") or os.environ.get("HOME")
if not operator_home:
    raise RuntimeError("set CAMELID_OPERATOR_HOME or HOME")
model_root = os.environ.get("CAMELID_MODEL_ROOT") or os.path.join(operator_home, "models")
PATH = os.environ.get("CAMELID_GEMMA4_CGHOST") or os.path.join(
    model_root, "gemma4-mtp-pair", "gemma-4-26B_q4_0-it.v3.cghost"
)
REC=3_345_408
F_NOCACHE=48
size=os.path.getsize(PATH)
nrec=size//REC
random.seed(1234)

def run(nthreads, nreads, nocache):
    offs=[random.randrange(0,nrec-1)*REC for _ in range(nreads)]
    fds=[]
    for _ in range(nthreads):
        fd=os.open(PATH, os.O_RDONLY)
        if nocache: fcntl.fcntl(fd, F_NOCACHE, 1)
        fds.append(fd)
    done=[0]*nthreads
    def worker(i):
        n=0
        for j in range(i,len(offs),nthreads):
            b=os.pread(fds[i],REC,offs[j]); n+=len(b)
        done[i]=n
    ts=[threading.Thread(target=worker,args=(i,)) for i in range(nthreads)]
    t0=time.monotonic(); [t.start() for t in ts]; [t.join() for t in ts]
    dt=time.monotonic()-t0
    for fd in fds: os.close(fd)
    tot=sum(done)
    return tot/dt/1e9, dt*1000, tot

print(f"file {PATH.split('/')[-1]}  {size/1e9:.1f} GB  records {nrec}")
print(f"{'threads':>8s} {'mode':>10s} {'reads':>6s} {'GB/s':>7s} {'ms':>8s}")
for nocache in (True,False):
    for nt in (1,2,4,8,16):
        bw,ms,tot=run(nt, 192, nocache)
        print(f"{nt:8d} {'F_NOCACHE' if nocache else 'cached':>10s} {192:6d} {bw:7.2f} {ms:8.1f}")
print()
print("engine MEASURED: 1.87 GB/s with CAMELID_GEMMA4_GHOST_READ_THREADS=4")
