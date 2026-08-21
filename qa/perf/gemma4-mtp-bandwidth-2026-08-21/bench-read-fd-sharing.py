#!/usr/bin/env python3
"""One shared fd (what the engine does) vs one fd per thread (what my bench did)."""
import os, time, random, threading, fcntl
PATH="/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost"
REC=3_345_408; F_NOCACHE=48
nrec=os.path.getsize(PATH)//REC
random.seed(11)

def run(batch, nthreads, shared_fd, batches=40):
    if shared_fd:
        fd=os.open(PATH, os.O_RDONLY); fcntl.fcntl(fd,F_NOCACHE,1); fds=[fd]*nthreads
    else:
        fds=[]
        for _ in range(nthreads):
            f=os.open(PATH, os.O_RDONLY); fcntl.fcntl(f,F_NOCACHE,1); fds.append(f)
    total=0; t0=time.monotonic()
    for _ in range(batches):
        offs=[random.randrange(0,nrec-1)*REC for _ in range(batch)]
        got=[0]*nthreads
        def w(i):
            n=0
            for j in range(i,len(offs),nthreads): n+=len(os.pread(fds[i],REC,offs[j]))
            got[i]=n
        ts=[threading.Thread(target=w,args=(i,)) for i in range(nthreads)]
        [t.start() for t in ts]; [t.join() for t in ts]
        total+=sum(got)
    dt=time.monotonic()-t0
    for f in set(fds): os.close(f)
    return total/dt/1e9

print(f"{'batch':>6s} {'threads':>8s} {'fd mode':>14s} {'GB/s':>7s}")
for batch in (7,28):
    for nt in (4,8):
        for shared,lbl in ((True,'SHARED (engine)'),(False,'per-thread')):
            print(f"{batch:6d} {nt:8d} {lbl:>14s} {run(batch,nt,shared):7.2f}")
    print()
