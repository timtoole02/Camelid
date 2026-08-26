#!/usr/bin/env python3
"""Take one non-terminating macOS safety sample for an ungoverned run.

This is deliberately a point sampler, not a watchdog: it never owns, signals,
or terminates the model process.  The no-watchdog runner invokes it before
launch, at readiness, and immediately after the request, then marks the
resulting receipt non-qualifying because nothing observes the intervals
between those samples.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from typing import NoReturn


SCHEMA_VERSION = 1
HOST_VM_INFO64 = 4
RUSAGE_INFO_V2 = 2
PROC_PGRP_ONLY = 2
MAX_PROCESS_GROUP_MEMBERS = 4096


class SampleError(RuntimeError):
    """Required point-in-time telemetry was unavailable or malformed."""


class VmStatistics64(ctypes.Structure):
    # /usr/include/mach/vm_statistics.h, vm_statistics64_data_t.
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
        ("swapped_count", ctypes.c_uint64),
    ]


class RusageInfoV2(ctypes.Structure):
    # /usr/include/sys/resource.h, rusage_info_v2.
    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        ("user_time", ctypes.c_uint64),
        ("system_time", ctypes.c_uint64),
        ("pkg_idle_wkups", ctypes.c_uint64),
        ("interrupt_wkups", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("wired_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("phys_footprint", ctypes.c_uint64),
        ("proc_start_abstime", ctypes.c_uint64),
        ("proc_exit_abstime", ctypes.c_uint64),
        ("child_user_time", ctypes.c_uint64),
        ("child_system_time", ctypes.c_uint64),
        ("child_pkg_idle_wkups", ctypes.c_uint64),
        ("child_interrupt_wkups", ctypes.c_uint64),
        ("child_pageins", ctypes.c_uint64),
        ("child_elapsed_abstime", ctypes.c_uint64),
        ("diskio_bytesread", ctypes.c_uint64),
        ("diskio_byteswritten", ctypes.c_uint64),
    ]


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        raise SampleError(f"kill(0) for PID {pid} failed: {error}") from error
    return True


def _process_error(pid: int, message: str) -> NoReturn:
    state = "still exists" if _pid_exists(pid) else "disappeared"
    raise SampleError(f"{message}; PID {pid} {state}")


class NativePointSampler:
    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise SampleError("manual safety sampling requires macOS")
        if ctypes.sizeof(VmStatistics64) != 160:
            raise SampleError(
                f"unexpected vm_statistics64 size {ctypes.sizeof(VmStatistics64)}"
            )
        if ctypes.sizeof(RusageInfoV2) != 160:
            raise SampleError(
                f"unexpected rusage_info_v2 size {ctypes.sizeof(RusageInfoV2)}"
            )

        self.lib = ctypes.CDLL(None, use_errno=True)
        self.lib.mach_host_self.argtypes = []
        self.lib.mach_host_self.restype = ctypes.c_uint32
        self.lib.host_page_size.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.lib.host_page_size.restype = ctypes.c_int
        self.lib.host_statistics64.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.lib.host_statistics64.restype = ctypes.c_int
        self.lib.sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.sysctlbyname.restype = ctypes.c_int
        self.lib.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.proc_pid_rusage.restype = ctypes.c_int
        self.lib.proc_listpids.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib.proc_listpids.restype = ctypes.c_int

        self.host = self.lib.mach_host_self()
        page_size = ctypes.c_uint32()
        result = self.lib.host_page_size(self.host, ctypes.byref(page_size))
        if result != 0 or page_size.value == 0:
            raise SampleError(f"host_page_size failed with Mach result {result}")
        self.page_size = int(page_size.value)
        self.vm_info_count = ctypes.sizeof(VmStatistics64) // ctypes.sizeof(
            ctypes.c_int32
        )

    def sample_host(self) -> dict[str, int]:
        vm = VmStatistics64()
        count = ctypes.c_uint32(self.vm_info_count)
        result = self.lib.host_statistics64(
            self.host,
            HOST_VM_INFO64,
            ctypes.cast(ctypes.byref(vm), ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(count),
        )
        if result != 0:
            raise SampleError(f"host_statistics64 failed with Mach result {result}")
        if count.value < self.vm_info_count:
            raise SampleError(
                f"host_statistics64 returned {count.value} integers, "
                f"expected at least {self.vm_info_count}"
            )

        pressure = ctypes.c_uint32()
        pressure_size = ctypes.c_size_t(ctypes.sizeof(pressure))
        ctypes.set_errno(0)
        result = self.lib.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            ctypes.byref(pressure),
            ctypes.byref(pressure_size),
            None,
            0,
        )
        if result != 0 or pressure_size.value != ctypes.sizeof(pressure):
            raise SampleError(
                "kern.memorystatus_vm_pressure_level failed: "
                f"result={result}, errno={ctypes.get_errno()}, "
                f"size={pressure_size.value}"
            )

        return {
            "page_size_bytes": self.page_size,
            "swapped_pages_current": int(vm.swapped_count),
            "swapped_bytes_current": int(vm.swapped_count) * self.page_size,
            "swapins_pages": int(vm.swapins),
            "swapouts_pages": int(vm.swapouts),
            "pressure_level_raw": int(pressure.value),
            "wired_bytes": int(vm.wire_count) * self.page_size,
        }

    def process_group_pids(self, process_group: int) -> list[int]:
        pids = (ctypes.c_int * MAX_PROCESS_GROUP_MEMBERS)()
        buffer_bytes = ctypes.sizeof(pids)
        ctypes.set_errno(0)
        result = self.lib.proc_listpids(
            PROC_PGRP_ONLY,
            process_group,
            ctypes.byref(pids),
            buffer_bytes,
        )
        if result < 0:
            raise SampleError(
                f"proc_listpids(PGRP {process_group}) failed: errno={ctypes.get_errno()}"
            )
        if result >= buffer_bytes or result % ctypes.sizeof(ctypes.c_int) != 0:
            raise SampleError(
                f"proc_listpids(PGRP {process_group}) returned malformed size {result}"
            )
        count = result // ctypes.sizeof(ctypes.c_int)
        return sorted({int(pid) for pid in pids[:count] if pid > 0})

    def sample_process(self, pid: int) -> dict[str, int]:
        for attempt in range(2):
            usage = RusageInfoV2()
            ctypes.set_errno(0)
            result = self.lib.proc_pid_rusage(
                pid, RUSAGE_INFO_V2, ctypes.byref(usage)
            )
            if result != 0:
                _process_error(
                    pid,
                    f"proc_pid_rusage failed: result={result}, errno={ctypes.get_errno()}",
                )
            if usage.resident_size > 0 and usage.phys_footprint > 0:
                return {
                    "rss_bytes": int(usage.resident_size),
                    "physical_footprint_bytes": int(usage.phys_footprint),
                }
            if not _pid_exists(pid):
                _process_error(pid, "proc_pid_rusage returned zero accounting")
            if attempt == 0:
                time.sleep(0.005)
        _process_error(pid, "proc_pid_rusage returned zero accounting twice")

    def sample_process_group(
        self, process_group: int, required_leader: int
    ) -> dict[str, object]:
        member_pids = self.process_group_pids(process_group)
        if required_leader not in member_pids:
            _process_error(
                required_leader,
                f"leader missing from process group {process_group}",
            )
        members = [self.sample_process(pid) for pid in member_pids]
        return {
            "pid": required_leader,
            "process_group": process_group,
            "member_pids": member_pids,
            "rss_bytes": sum(member["rss_bytes"] for member in members),
            "physical_footprint_bytes": sum(
                member["physical_footprint_bytes"] for member in members
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("pre", "ready", "post"))
    parser.add_argument("--pid", type=int)
    parser.add_argument("--process-group", type=int)
    args = parser.parse_args()

    wants_process = args.phase != "pre"
    has_pid = args.pid is not None
    has_process_group = args.process_group is not None
    if (wants_process and not (has_pid and has_process_group)) or (
        not wants_process and (has_pid or has_process_group)
    ):
        parser.error("ready/post require --pid and --process-group; pre forbids them")
    if wants_process and (args.pid <= 0 or args.process_group <= 0):
        parser.error("PID and process group must be positive")

    sampler = NativePointSampler()
    process = None
    if wants_process:
        process = sampler.sample_process_group(args.process_group, args.pid)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": args.phase,
        "host": sampler.sample_host(),
        "process": process,
    }
    json.dump(payload, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SampleError as error:
        print(f"manual safety sample failed: {error}", file=sys.stderr)
        raise SystemExit(75) from error
