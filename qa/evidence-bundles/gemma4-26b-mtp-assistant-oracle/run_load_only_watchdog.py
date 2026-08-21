#!/usr/bin/env python3
"""Fail-closed macOS watchdog for the Gemma 4 load-only residency probe.

The child must be a previously built, internal-APFS test binary.  This wrapper
does not invoke Cargo and deliberately samples Mach/libproc directly instead
of forking ``vm_stat`` and ``sysctl`` four times per second.

Example (all paths must be on the system data volume)::

    python3 run_load_only_watchdog.py \
      --report /absolute/run/load-only-report.json \
      --watchdog-log /absolute/run/watchdog.jsonl \
      --child-log /absolute/run/load-only.log \
      -- /absolute/run/gemma4_mtp_assistant_experiment-HASH \
         gemma4_mtp_assistant_load_only_probe \
         --ignored --exact --nocapture --test-threads=1

The report is owned by the child.  The watchdog never opens, truncates,
removes, or replaces it, so an atomic checkpoint published before a kill is
left intact.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from typing import Any, NoReturn, Sequence
from urllib.parse import unquote, urlparse


SCHEMA_VERSION = 2
SAMPLE_PERIOD_NS = 250_000_000
TERM_GRACE_NS = 250_000_000
MIN_RECLAIMABLE_HEADROOM_BYTES = 2 * 1024 * 1024 * 1024
NORMAL_PRESSURE_LEVEL = 1
HOST_VM_INFO64 = 4
RUSAGE_INFO_V2 = 2

EXIT_WATCHDOG_ABORT = 86
EXIT_BASELINE_REFUSED = 75
EXIT_IO_ERROR = 74
EXIT_SOFTWARE_ERROR = 70

LOAD_ONLY_ENABLE_ENV = "CAMELID_GEMMA4_MTP_LOAD_ONLY_PROBE"
LOAD_ONLY_REPORT_ENV = "CAMELID_GEMMA4_MTP_LOAD_ONLY_REPORT_PATH"
SYSTEM_CHILD_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


class PreflightError(RuntimeError):
    """The requested run is not isolated enough to start."""


class TelemetryError(RuntimeError):
    """Required fail-closed telemetry was unavailable or malformed."""


class ParentSignal(RuntimeError):
    """The watchdog parent received a signal that must reach the child."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"watchdog parent received signal {signum}")
        self.signum = signum


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


class NativeTelemetry:
    """Allocation-free wrappers around the macOS host and libproc APIs."""

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise TelemetryError("the load-only watchdog requires macOS")
        if ctypes.sizeof(VmStatistics64) != 160:
            raise TelemetryError(
                f"unexpected vm_statistics64 size {ctypes.sizeof(VmStatistics64)}"
            )
        if ctypes.sizeof(RusageInfoV2) != 160:
            raise TelemetryError(
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

        self.host = self.lib.mach_host_self()
        page_size = ctypes.c_uint32()
        result = self.lib.host_page_size(self.host, ctypes.byref(page_size))
        if result != 0 or page_size.value == 0:
            raise TelemetryError(f"host_page_size failed with Mach result {result}")
        self.page_size = int(page_size.value)
        self.vm_info_count = ctypes.sizeof(VmStatistics64) // ctypes.sizeof(
            ctypes.c_int32
        )

    def sample_host(self) -> dict[str, int]:
        started_ns = time.monotonic_ns()
        vm = VmStatistics64()
        count = ctypes.c_uint32(self.vm_info_count)
        result = self.lib.host_statistics64(
            self.host,
            HOST_VM_INFO64,
            ctypes.cast(ctypes.byref(vm), ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(count),
        )
        if result != 0:
            raise TelemetryError(
                f"host_statistics64 failed with Mach result {result}"
            )
        if count.value < self.vm_info_count:
            raise TelemetryError(
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
            err = ctypes.get_errno()
            raise TelemetryError(
                "kern.memorystatus_vm_pressure_level failed: "
                f"result={result}, errno={err}, size={pressure_size.value}"
            )

        # host_statistics64's free_count includes speculative pages on Darwin
        # 25, whereas `vm_stat` reports "Pages free" with them removed. Preserve
        # that strict value separately; reclaimable headroom adds inactive pages.
        if vm.speculative_count > vm.free_count:
            raise TelemetryError(
                "vm_statistics64 speculative_count exceeds free_count"
            )
        strict_free_pages = int(vm.free_count - vm.speculative_count)
        free_bytes_strict = strict_free_pages * self.page_size
        inactive_bytes = int(vm.inactive_count) * self.page_size
        observed_ns = time.monotonic_ns()
        return {
            "sample_started_monotonic_ns": started_ns,
            "observed_monotonic_ns": observed_ns,
            "sample_duration_ns": observed_ns - started_ns,
            "unix_time_ns": time.time_ns(),
            "page_size_bytes": self.page_size,
            "free_pages_raw_including_speculative": int(vm.free_count),
            "speculative_pages": int(vm.speculative_count),
            "free_pages_strict": strict_free_pages,
            "free_bytes_strict": free_bytes_strict,
            "active_bytes": int(vm.active_count) * self.page_size,
            "inactive_bytes": inactive_bytes,
            "reclaimable_headroom_bytes": free_bytes_strict + inactive_bytes,
            "wired_bytes": int(vm.wire_count) * self.page_size,
            "compressor_occupied_bytes": int(vm.compressor_page_count)
            * self.page_size,
            "compressed_logical_bytes": int(
                vm.total_uncompressed_pages_in_compressor
            )
            * self.page_size,
            "pageins_bytes": int(vm.pageins) * self.page_size,
            "pageouts_bytes": int(vm.pageouts) * self.page_size,
            "swapins_pages": int(vm.swapins),
            "swapins_bytes": int(vm.swapins) * self.page_size,
            "swapouts_pages": int(vm.swapouts),
            "swapouts_bytes": int(vm.swapouts) * self.page_size,
            "swapped_pages_current": int(vm.swapped_count),
            "swapped_bytes_current": int(vm.swapped_count) * self.page_size,
            "pressure_level_raw": int(pressure.value),
        }

    def sample_process(self, pid: int) -> dict[str, int]:
        usage = RusageInfoV2()
        ctypes.set_errno(0)
        result = self.lib.proc_pid_rusage(
            pid, RUSAGE_INFO_V2, ctypes.byref(usage)
        )
        if result != 0:
            err = ctypes.get_errno()
            raise TelemetryError(
                f"proc_pid_rusage({pid}) failed: result={result}, errno={err}"
            )
        if usage.resident_size == 0 or usage.phys_footprint == 0:
            raise TelemetryError(
                f"proc_pid_rusage({pid}) returned zero resident/footprint bytes"
            )
        return {
            "pid": pid,
            "rss_bytes": int(usage.resident_size),
            "physical_footprint_bytes": int(usage.phys_footprint),
            "wired_bytes": int(usage.wired_size),
            "pageins": int(usage.pageins),
            "disk_read_bytes": int(usage.diskio_bytesread),
            "disk_written_bytes": int(usage.diskio_byteswritten),
        }


class JsonlLog:
    def __init__(self, path: Path) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(path, flags, 0o600)
        self.path = path
        self.file = os.fdopen(fd, "w", encoding="utf-8", buffering=1)

    def write(self, event: dict[str, Any]) -> None:
        event = {
            "schema_version": SCHEMA_VERSION,
            **event,
        }
        self.file.write(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()


def _die(message: str, exit_code: int = 2) -> NoReturn:
    print(f"load-only watchdog: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _path_is_on_internal_system_volume(path: Path) -> bool:
    root_device = os.stat("/").st_dev
    return os.stat(path).st_dev == root_device


def _require_internal_existing(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"{label} does not resolve: {path}: {error}") from error
    if str(resolved) == "/Volumes" or str(resolved).startswith("/Volumes/"):
        raise PreflightError(f"{label} resolves under /Volumes: {resolved}")
    if not _path_is_on_internal_system_volume(resolved):
        raise PreflightError(f"{label} is not on the system data volume: {resolved}")
    return resolved


def _require_new_internal_output(raw: str, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise PreflightError(f"{label} must be absolute: {path}")
    if os.path.lexists(path):
        raise PreflightError(f"{label} already exists: {path}")
    parent = _require_internal_existing(path.parent, f"{label} parent")
    if not parent.is_dir():
        raise PreflightError(f"{label} parent is not a directory: {parent}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise PreflightError(f"{label} parent is not writable: {parent}")
    return parent / path.name


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path
    while not os.path.lexists(candidate):
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def _check_path_candidate(raw: str, label: str) -> None:
    if not raw.startswith("/"):
        return
    path = Path(raw)
    existing = _nearest_existing_ancestor(path)
    _require_internal_existing(existing, label)
    if os.path.lexists(path):
        _require_internal_existing(path, label)


def _preflight_child_argument(raw: str, index: int) -> None:
    if "\x00" in raw:
        raise PreflightError(f"child argument {index} contains NUL")
    if raw == "/Volumes" or "/Volumes/" in raw:
        raise PreflightError(f"child argument {index} names an external volume: {raw}")

    candidates = [raw]
    if "=" in raw:
        candidates.append(raw.split("=", 1)[1])
    for candidate in candidates:
        if candidate.startswith("file://"):
            parsed = urlparse(candidate)
            if parsed.netloc not in ("", "localhost"):
                raise PreflightError(
                    f"child argument {index} has a non-local file URI: {raw}"
                )
            candidate = unquote(parsed.path)
        _check_path_candidate(candidate, f"child argument {index}")


def _preflight_environment(environment: dict[str, str]) -> None:
    # Path lookup inside the child is limited to Apple's internal system tools.
    environment["PATH"] = SYSTEM_CHILD_PATH
    environment.pop("CDPATH", None)
    for key in list(environment):
        if key.startswith("DYLD_"):
            environment.pop(key)

    temp_dir = environment.get("TMPDIR", "/tmp")
    _require_internal_existing(Path(temp_dir), "child TMPDIR")
    environment["TMPDIR"] = temp_dir

    # The experiment's file inputs are environment-backed.  Catch literal T7
    # paths and internal symlinks that resolve to any mounted external volume.
    for key, value in environment.items():
        if key.startswith("CAMELID_") or key.startswith("SPEC50_"):
            if value == "/Volumes" or "/Volumes/" in value:
                raise PreflightError(f"child environment {key} names /Volumes: {value}")
            if value.startswith("/"):
                _check_path_candidate(value, f"child environment {key}")


def _preflight_command(command: Sequence[str]) -> list[str]:
    if not command:
        raise PreflightError("a child command is required after --")
    executable = Path(command[0])
    if not executable.is_absolute():
        raise PreflightError("the child executable must be an absolute path")
    executable = _require_internal_existing(executable, "child executable")
    mode = executable.stat().st_mode
    if not stat.S_ISREG(mode) or not os.access(executable, os.X_OK):
        raise PreflightError(f"child executable is not an executable file: {executable}")
    if executable.name in {"cargo", "rustc", "rustup"}:
        raise PreflightError(f"refusing build tool as timed child: {executable}")

    result = [str(executable), *command[1:]]
    for index, argument in enumerate(result):
        _preflight_child_argument(argument, index)
    return result


def _open_exclusive_child_log(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, "wb", buffering=0)


def _host_violation_reasons(
    sample: dict[str, int], baseline_swapouts_pages: int
) -> list[str]:
    reasons: list[str] = []
    if sample["swapouts_pages"] > baseline_swapouts_pages:
        reasons.append(
            "swapouts_increased_by_"
            f"{sample['swapouts_pages'] - baseline_swapouts_pages}_pages"
        )
    if sample["pressure_level_raw"] != NORMAL_PRESSURE_LEVEL:
        reasons.append(f"pressure_level_{sample['pressure_level_raw']}_is_not_normal")
    if sample["reclaimable_headroom_bytes"] < MIN_RECLAIMABLE_HEADROOM_BYTES:
        reasons.append(
            "reclaimable_headroom_bytes_"
            f"{sample['reclaimable_headroom_bytes']}_below_"
            f"{MIN_RECLAIMABLE_HEADROOM_BYTES}"
        )
    return reasons


def _send_group_signal(child: subprocess.Popen[bytes], signum: int) -> bool:
    if child.poll() is not None:
        return False
    try:
        os.killpg(child.pid, signum)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        # start_new_session=True should make pid==pgid.  A direct-PID fallback
        # still prevents an allocating test process from escaping on anomaly.
        try:
            os.kill(child.pid, signum)
            return True
        except ProcessLookupError:
            return False


def _finish_term_then_kill(
    child: subprocess.Popen[bytes], term_sent_ns: int
) -> int:
    deadline_ns = term_sent_ns + TERM_GRACE_NS
    while child.poll() is None and time.monotonic_ns() < deadline_ns:
        time.sleep(0.01)
    if child.poll() is None:
        _send_group_signal(child, signal.SIGKILL)
    try:
        return child.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        _send_group_signal(child, signal.SIGKILL)
        return child.wait()


def _portable_child_exit(returncode: int) -> int:
    if returncode >= 0:
        return min(returncode, 255)
    return min(128 + (-returncode), 255)


def _report_metadata(report: Path) -> dict[str, Any]:
    try:
        metadata = report.lstat()
    except FileNotFoundError:
        return {"report_exists": False}
    return {
        "report_exists": True,
        "report_size_bytes": metadata.st_size,
        "report_mtime_ns": metadata.st_mtime_ns,
        "report_is_regular_file": stat.S_ISREG(metadata.st_mode),
        "report_is_symlink": stat.S_ISLNK(metadata.st_mode),
    }


def _install_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> NoReturn:
        raise ParentSignal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.signal(signum, handler)
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _run_self_test() -> int:
    telemetry = NativeTelemetry()
    first = telemetry.sample_host()
    process = telemetry.sample_process(os.getpid())
    second = telemetry.sample_host()
    if second["swapouts_pages"] < first["swapouts_pages"]:
        raise TelemetryError("swapouts counter moved backwards during self-test")
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "self_test": "pass",
                "host": second,
                "process": process,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def run(args: argparse.Namespace) -> int:
    report = _require_new_internal_output(args.report, "report")
    watchdog_path = _require_new_internal_output(args.watchdog_log, "watchdog log")
    child_log_path = _require_new_internal_output(args.child_log, "child log")
    if len({report, watchdog_path, child_log_path}) != 3:
        raise PreflightError("report, watchdog log, and child log must be distinct")
    _require_internal_existing(Path.cwd(), "watchdog working directory")

    command = list(args.child_command)
    if not command or command[0] != "--":
        raise PreflightError("put the direct child command after a literal --")
    command = _preflight_command(command[1:])

    environment = dict(os.environ)
    existing_report = environment.get(LOAD_ONLY_REPORT_ENV)
    if existing_report is not None and Path(existing_report) != report:
        raise PreflightError(
            f"{LOAD_ONLY_REPORT_ENV} disagrees with --report: {existing_report}"
        )
    existing_enable = environment.get(LOAD_ONLY_ENABLE_ENV)
    if existing_enable is not None and existing_enable != "1":
        raise PreflightError(f"{LOAD_ONLY_ENABLE_ENV} must be 1 when set")
    environment[LOAD_ONLY_REPORT_ENV] = str(report)
    environment[LOAD_ONLY_ENABLE_ENV] = "1"
    environment["PWD"] = str(Path.cwd())
    _preflight_environment(environment)

    telemetry = NativeTelemetry()
    watchdog_log: JsonlLog | None = None
    child_log = None
    child: subprocess.Popen[bytes] | None = None
    previous_handlers: dict[int, Any] = {}
    min_free_bytes = 2**63 - 1
    min_reclaimable_headroom_bytes = 2**63 - 1
    peak_rss_bytes = 0
    peak_footprint_bytes = 0
    baseline_swapouts_pages = 0
    sequence = 0

    try:
        watchdog_log = JsonlLog(watchdog_path)
        child_log = _open_exclusive_child_log(child_log_path)
        baseline = telemetry.sample_host()
        baseline_swapouts_pages = baseline["swapouts_pages"]
        min_free_bytes = baseline["free_bytes_strict"]
        min_reclaimable_headroom_bytes = baseline[
            "reclaimable_headroom_bytes"
        ]
        baseline_reasons = _host_violation_reasons(
            baseline, baseline_swapouts_pages
        )
        if baseline["sample_duration_ns"] >= SAMPLE_PERIOD_NS:
            baseline_reasons.append("baseline_telemetry_exceeded_250ms")
        watchdog_log.write(
            {
                "event": "clean_parent_baseline",
                "sequence": sequence,
                "host": baseline,
                "violations": baseline_reasons,
            }
        )
        if baseline_reasons:
            watchdog_log.write(
                {
                    "event": "baseline_refused",
                    "sequence": sequence,
                    "violations": baseline_reasons,
                    **_report_metadata(report),
                }
            )
            return EXIT_BASELINE_REFUSED

        # Close the last race with a stale producer after reserving both logs.
        if os.path.lexists(report):
            raise PreflightError(f"report appeared before child spawn: {report}")

        watched_signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
        previous_handlers = _install_signal_handlers()
        spawn_started_ns = time.monotonic_ns()
        try:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=child_log,
                stderr=subprocess.STDOUT,
                cwd=Path.cwd(),
                env=environment,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            # A pending parent signal is delivered only after Popen has either
            # returned an assigned child handle or failed without a child.
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        try:
            process_group = os.getpgid(child.pid)
        except ProcessLookupError:
            process_group = child.pid
        if child.poll() is None and process_group != child.pid:
            raise TelemetryError(
                f"child process group {process_group} does not equal PID {child.pid}"
            )
        watchdog_log.write(
            {
                "event": "child_started",
                "sequence": sequence,
                "pid": child.pid,
                "process_group": process_group,
                "command": command,
                "sample_period_ns": SAMPLE_PERIOD_NS,
                "reclaimable_headroom_formula": (
                    "free_bytes_strict_plus_inactive_bytes"
                ),
                "minimum_reclaimable_headroom_bytes": (
                    MIN_RECLAIMABLE_HEADROOM_BYTES
                ),
                "baseline_swapouts_pages": baseline_swapouts_pages,
                "report": str(report),
            }
        )

        # The first safety deadline starts at spawn, not after the durable
        # child-start record.  A slow log fsync must therefore fail closed
        # instead of creating an unobserved allocation window.
        next_due_ns = spawn_started_ns
        watchdog_aborted = False
        abort_reasons: list[str] = []
        child_returncode: int | None = None

        while True:
            now_ns = time.monotonic_ns()
            if now_ns < next_due_ns:
                time.sleep((next_due_ns - now_ns) / 1_000_000_000)
            sample_started_ns = time.monotonic_ns()
            schedule_lateness_ns = max(0, sample_started_ns - next_due_ns)
            sequence += 1

            try:
                host = telemetry.sample_host()
                process: dict[str, int] | None = None
                if child.poll() is None:
                    try:
                        process = telemetry.sample_process(child.pid)
                    except TelemetryError:
                        # ESRCH is an ordinary race only after waitpid confirms
                        # that the direct child has exited.
                        if child.poll() is None:
                            raise
                telemetry_finished_ns = time.monotonic_ns()
            except TelemetryError as error:
                watchdog_aborted = True
                abort_reasons = [f"telemetry_failure:{error}"]
                term_sent_ns = time.monotonic_ns()
                _send_group_signal(child, signal.SIGTERM)
                watchdog_log.write(
                    {
                        "event": "watchdog_abort",
                        "sequence": sequence,
                        "pid": child.pid,
                        "violations": abort_reasons,
                        "term_sent_monotonic_ns": term_sent_ns,
                        **_report_metadata(report),
                    }
                )
                child_returncode = _finish_term_then_kill(child, term_sent_ns)
                break

            telemetry_duration_ns = telemetry_finished_ns - sample_started_ns
            min_free_bytes = min(min_free_bytes, host["free_bytes_strict"])
            min_reclaimable_headroom_bytes = min(
                min_reclaimable_headroom_bytes,
                host["reclaimable_headroom_bytes"],
            )
            if process is not None:
                peak_rss_bytes = max(peak_rss_bytes, process["rss_bytes"])
                peak_footprint_bytes = max(
                    peak_footprint_bytes, process["physical_footprint_bytes"]
                )
            reasons = _host_violation_reasons(host, baseline_swapouts_pages)
            if telemetry_duration_ns + schedule_lateness_ns >= SAMPLE_PERIOD_NS:
                reasons.append(
                    "telemetry_overrun:"
                    f"lateness={schedule_lateness_ns},duration={telemetry_duration_ns}"
                )
            sample_event = {
                "event": "sample",
                "sequence": sequence,
                "pid": child.pid,
                "scheduled_monotonic_ns": next_due_ns,
                "schedule_lateness_ns": schedule_lateness_ns,
                "telemetry_duration_ns": telemetry_duration_ns,
                "host": host,
                "process": process,
                "violations": reasons,
            }
            if reasons:
                watchdog_aborted = True
                abort_reasons = reasons
                # Stop allocation first.  Evidence is then flushed/fsynced
                # during the short TERM grace before a compulsory KILL.
                term_sent_ns = time.monotonic_ns()
                _send_group_signal(child, signal.SIGTERM)
                sample_event["event"] = "watchdog_abort"
                sample_event["term_sent_monotonic_ns"] = term_sent_ns
                sample_event.update(_report_metadata(report))
                watchdog_log.write(sample_event)
                child_returncode = _finish_term_then_kill(child, term_sent_ns)
                break

            watchdog_log.write(sample_event)
            durable_sample_finished_ns = time.monotonic_ns()
            durable_sample_duration_ns = (
                durable_sample_finished_ns - sample_started_ns
            )
            if durable_sample_duration_ns + schedule_lateness_ns >= SAMPLE_PERIOD_NS:
                watchdog_aborted = True
                abort_reasons = [
                    "durable_sample_overrun:"
                    f"lateness={schedule_lateness_ns},"
                    f"duration={durable_sample_duration_ns}"
                ]
                term_sent_ns = time.monotonic_ns()
                _send_group_signal(child, signal.SIGTERM)
                watchdog_log.write(
                    {
                        "event": "watchdog_abort",
                        "sequence": sequence,
                        "pid": child.pid,
                        "violations": abort_reasons,
                        "term_sent_monotonic_ns": term_sent_ns,
                        "durable_sample_duration_ns": durable_sample_duration_ns,
                        **_report_metadata(report),
                    }
                )
                child_returncode = _finish_term_then_kill(child, term_sent_ns)
                break
            child_returncode = child.poll()
            if child_returncode is not None:
                # Close the interval between the last live sample and waitpid.
                final_host = telemetry.sample_host()
                min_free_bytes = min(
                    min_free_bytes, final_host["free_bytes_strict"]
                )
                min_reclaimable_headroom_bytes = min(
                    min_reclaimable_headroom_bytes,
                    final_host["reclaimable_headroom_bytes"],
                )
                final_reasons = _host_violation_reasons(
                    final_host, baseline_swapouts_pages
                )
                if final_host["sample_duration_ns"] >= SAMPLE_PERIOD_NS:
                    final_reasons.append("final_telemetry_exceeded_250ms")
                watchdog_log.write(
                    {
                        "event": "post_exit_sample",
                        "sequence": sequence + 1,
                        "pid": child.pid,
                        "host": final_host,
                        "violations": final_reasons,
                    }
                )
                if final_reasons:
                    watchdog_aborted = True
                    abort_reasons = final_reasons
                break

            next_due_ns += SAMPLE_PERIOD_NS

        report_metadata = _report_metadata(report)
        watchdog_log.write(
            {
                "event": "final",
                "sequence": sequence + 2,
                "pid": child.pid,
                "child_returncode": child_returncode,
                "watchdog_aborted": watchdog_aborted,
                "abort_reasons": abort_reasons,
                "minimum_free_bytes_observed": min_free_bytes,
                "minimum_reclaimable_headroom_bytes_observed": (
                    min_reclaimable_headroom_bytes
                ),
                "peak_child_rss_bytes": peak_rss_bytes,
                "peak_child_physical_footprint_bytes": peak_footprint_bytes,
                "baseline_swapouts_pages": baseline_swapouts_pages,
                **report_metadata,
            }
        )
        if watchdog_aborted:
            return EXIT_WATCHDOG_ABORT
        if child_returncode is None:
            return EXIT_SOFTWARE_ERROR
        if child_returncode == 0 and not report_metadata["report_exists"]:
            return EXIT_IO_ERROR
        return _portable_child_exit(child_returncode)

    except ParentSignal as caught:
        if child is not None:
            # Prevent repeated user signals from interrupting the mandatory
            # TERM/KILL cleanup path.
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(signum, signal.SIG_IGN)
            term_sent_ns = time.monotonic_ns()
            _send_group_signal(child, signal.SIGTERM)
            if watchdog_log is not None:
                watchdog_log.write(
                    {
                        "event": "parent_signal_abort",
                        "sequence": sequence + 1,
                        "pid": child.pid,
                        "signal": caught.signum,
                        "term_sent_monotonic_ns": term_sent_ns,
                        **_report_metadata(report),
                    }
                )
            _finish_term_then_kill(child, term_sent_ns)
        return min(128 + caught.signum, 255)
    except Exception as error:
        if child is not None and child.poll() is None:
            term_sent_ns = time.monotonic_ns()
            _send_group_signal(child, signal.SIGTERM)
            try:
                if watchdog_log is not None:
                    watchdog_log.write(
                        {
                            "event": "unexpected_failure_abort",
                            "sequence": sequence + 1,
                            "pid": child.pid,
                            "error": f"{type(error).__name__}: {error}",
                            "term_sent_monotonic_ns": term_sent_ns,
                            **_report_metadata(report),
                        }
                    )
            finally:
                _finish_term_then_kill(child, term_sent_ns)
        raise
    finally:
        if previous_handlers:
            _restore_signal_handlers(previous_handlers)
        if child_log is not None:
            try:
                child_log.flush()
                os.fsync(child_log.fileno())
            finally:
                child_log.close()
        if watchdog_log is not None:
            watchdog_log.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated Gemma 4 load-only probe under a 250 ms "
        "fail-closed macOS memory watchdog."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="sample this process and the host without creating files or spawning a child",
    )
    parser.add_argument("--report", help="absolute fresh child checkpoint path")
    parser.add_argument("--watchdog-log", help="absolute fresh watchdog JSONL path")
    parser.add_argument("--child-log", help="absolute fresh child stdout/stderr path")
    parser.add_argument("child_command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.self_test:
        if any(
            value is not None
            for value in (args.report, args.watchdog_log, args.child_log)
        ) or args.child_command:
            _die("--self-test cannot be combined with output paths or a child command")
        return _run_self_test()
    missing = [
        name
        for name, value in (
            ("--report", args.report),
            ("--watchdog-log", args.watchdog_log),
            ("--child-log", args.child_log),
        )
        if value is None
    ]
    if missing:
        _die(f"missing required arguments: {', '.join(missing)}")
    try:
        return run(args)
    except PreflightError as error:
        _die(str(error))
    except TelemetryError as error:
        _die(f"telemetry unavailable before spawn: {error}", EXIT_SOFTWARE_ERROR)
    except OSError as error:
        _die(f"I/O failure: {error}", EXIT_IO_ERROR)
    except Exception as error:
        _die(f"unexpected failure: {type(error).__name__}: {error}", EXIT_SOFTWARE_ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
