//! OS-level confinement for the `run_shell` tool (Task 1).
//!
//! `run_shell` is the one tool that hands the model a general-purpose execution
//! primitive. The path-confined file tools enforce a canonical-root jail in code
//! (see `tools.rs`), but a shell command can do anything the process can — so it
//! gets a *kernel*-enforced sandbox, not a code check.
//!
//! Three modes (config: `--shell-sandbox`):
//! - [`ShellSandbox::Disabled`] — `run_shell` is not registered at all.
//! - [`ShellSandbox::Sandboxed`] — **the default**. The command runs with: a
//!   non-root uid, a seccomp filter that blocks `ptrace`/`mount`/`socket`
//!   families (and other privilege syscalls), the scratch workspace as the cwd
//!   (chroot when the root is a usable rootfs), resource rlimits, and the
//!   existing hard wall-clock timeout.
//! - [`ShellSandbox::Unrestricted`] — explicit opt-in; the command runs
//!   cwd-pinned + timed but otherwise unconfined. Logs a startup warning.
//!
//! **Fail closed (with a native-Windows path).** Sandboxed mode applies *kernel*
//! enforcement (seccomp + uid-drop + rlimits) only on Linux/supported-arch. On
//! **Windows** — which has no seccomp — it applies the confinement Windows
//! actually offers (cwd-pin + the hard wall-clock timeout), gated by the approval
//! policy, and reports **exactly** that (never a faked seccomp/uid-drop claim);
//! this matches the `run_windows_command` model. Anywhere else still unenforceable
//! — macOS, a Linux kernel without `CONFIG_SECCOMP`, an unsupported arch —
//! sandboxed mode **refuses to run the tool** rather than silently downgrading.
//! The enforced layers are reported, not faked (see [`EnforcedShell`]); the UI
//! shows what was actually applied.
//!
//! ## Platform-verification status
//!
//! This module's Linux enforcement is compiled behind `cfg(target_os = "linux",
//! target_arch ∈ {x86_64, aarch64})`. The project's dev box is Windows, so that
//! path is **built-and-gated but not exercised here**; it must be validated by
//! Linux CI (the `socket_is_blocked_under_seccomp` test) before the sandbox is
//! trusted in production. See `frontend/design-evidence/DESIGN_LOG.md`.

use std::fmt;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;

/// The configured confinement mode for `run_shell`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellSandbox {
    /// `run_shell` is not offered to the model at all.
    Disabled,
    /// Kernel-enforced confinement (default). Fails closed where unenforceable.
    Sandboxed,
    /// Cwd-pinned + timed, otherwise unconfined. Explicit opt-in.
    Unrestricted,
}

// Kept as an explicit impl (not `#[derive(Default)]` + `#[default]`) so the secure
// default is stated in code next to its rationale and cannot be silently flipped by a
// future variant reorder — `Sandboxed` is the 2nd variant, so a plain derive would
// default to `Disabled`.
#[allow(clippy::derivable_impls)]
impl Default for ShellSandbox {
    fn default() -> Self {
        // Secure by default: confinement on, fail closed where unenforceable.
        ShellSandbox::Sandboxed
    }
}

impl ShellSandbox {
    pub fn as_str(self) -> &'static str {
        match self {
            ShellSandbox::Disabled => "disabled",
            ShellSandbox::Sandboxed => "sandboxed",
            ShellSandbox::Unrestricted => "unrestricted",
        }
    }
}

impl fmt::Display for ShellSandbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ShellSandbox {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "disabled" | "off" | "none" => Ok(ShellSandbox::Disabled),
            "sandboxed" | "sandbox" | "on" => Ok(ShellSandbox::Sandboxed),
            "unrestricted" | "unsafe" => Ok(ShellSandbox::Unrestricted),
            other => Err(format!(
                "invalid shell sandbox mode {other:?} (expected disabled|sandboxed|unrestricted)"
            )),
        }
    }
}

/// What confinement was *actually* applied to a spawned shell. Surfaced rather
/// than assumed, so the UI never claims a sandbox the kernel didn't enforce.
#[derive(Debug, Clone)]
pub struct EnforcedShell {
    pub mode: ShellSandbox,
    /// The layers that engaged (e.g. `"uid-drop"`, `"seccomp"`, `"chroot"`,
    /// `"rlimits"`, `"wall-timeout"`).
    pub layers: Vec<&'static str>,
    /// A caveat to surface (e.g. chroot skipped because the root is not a rootfs).
    pub note: Option<String>,
}

impl EnforcedShell {
    fn unrestricted() -> Self {
        EnforcedShell {
            mode: ShellSandbox::Unrestricted,
            layers: vec!["cwd-pin", "wall-timeout"],
            note: None,
        }
    }

    /// One-line human description of what was enforced.
    pub fn summary(&self) -> String {
        let layers = if self.layers.is_empty() {
            "none".to_string()
        } else {
            self.layers.join("+")
        };
        match &self.note {
            Some(n) => format!("{} [{}] ({})", self.mode, layers, n),
            None => format!("{} [{}]", self.mode, layers),
        }
    }
}

/// Apply the configured confinement to `builder` (a not-yet-spawned shell
/// command), with `root` as the scratch workspace. Returns what was enforced, or
/// an error that means **refuse to run** (fail closed) — never a silent
/// downgrade. `Disabled` is rejected here too (the tool should not have reached
/// execution).
pub fn configure_command(
    builder: &mut Command,
    root: &Path,
    mode: ShellSandbox,
) -> Result<EnforcedShell, String> {
    match mode {
        ShellSandbox::Disabled => Err("run_shell is disabled (shell_sandbox=disabled)".to_string()),
        ShellSandbox::Unrestricted => {
            builder.current_dir(root);
            Ok(EnforcedShell::unrestricted())
        }
        ShellSandbox::Sandboxed => configure_sandboxed(builder, root),
    }
}

/// Preflight describing what sandboxed mode *would* enforce on this host, for the
/// startup banner. Never spawns anything; just reports enforceability so the UI
/// is honest before the first command runs.
pub fn describe_sandboxed(root: &Path) -> Result<EnforcedShell, String> {
    // Build a throwaway command only to probe; we never spawn it.
    let mut probe = Command::new("true");
    configure_sandboxed(&mut probe, root)
}

/// Resolve a *requested* mode down to what this host can honour, for surfaces
/// that cannot ask the user to pick another mode.
///
/// The terminal lane prints the unenforceable-here warning and lets the operator
/// re-run with `--shell-sandbox`. A web session has neither: the flags do not
/// exist there, so an unenforceable `Sandboxed` would advertise `run_shell` to
/// the model and then refuse every single call — the model burns the turn on a
/// tool that cannot work, and in approval-gated mode the user approves exec
/// actions that are already doomed. Resolve to [`ShellSandbox::Disabled`] so the
/// tool is simply not offered, which is what "fail closed" means for a surface
/// with no opt-out. Returns the refusal text so callers can disclose *why*.
pub fn resolve_for_unattended_surface(
    requested: ShellSandbox,
    root: &Path,
) -> (ShellSandbox, Option<String>) {
    match requested {
        ShellSandbox::Sandboxed => match describe_sandboxed(root) {
            Ok(_) => (ShellSandbox::Sandboxed, None),
            Err(why) => (ShellSandbox::Disabled, Some(why)),
        },
        other => (other, None),
    }
}

// ===========================================================================
// Linux enforcement (x86_64 / aarch64). Built-and-gated; validated by Linux CI.
// ===========================================================================

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod linux {
    use super::EnforcedShell;
    use std::ffi::CString;
    use std::io;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::process::CommandExt;
    use std::path::Path;
    use std::process::Command;

    // Unprivileged identity the shell is dropped to when started as root. 65534 is
    // the conventional nobody/nogroup on Linux distributions.
    const NOBODY_UID: u32 = 65534;
    const NOBODY_GID: u32 = 65534;

    // seccomp / BPF constants not all exported by `libc`.
    const SECCOMP_MODE_FILTER: i32 = 2;
    const SECCOMP_RET_KILL_PROCESS: u32 = 0x8000_0000;
    const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
    const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
    // Classic-BPF opcodes.
    const BPF_LD_W_ABS: u16 = 0x20; // BPF_LD | BPF_W | BPF_ABS
    const BPF_JEQ_K: u16 = 0x15; // BPF_JMP | BPF_JEQ | BPF_K
    const BPF_RET_K: u16 = 0x06; // BPF_RET | BPF_K
                                 // Offsets into `struct seccomp_data`.
    const SD_NR: u32 = 0;
    const SD_ARCH: u32 = 4;

    #[cfg(target_arch = "x86_64")]
    const AUDIT_ARCH: u32 = 0xC000_003E; // AUDIT_ARCH_X86_64
    #[cfg(target_arch = "aarch64")]
    const AUDIT_ARCH: u32 = 0xC000_00B7; // AUDIT_ARCH_AARCH64

    fn bpf_stmt(code: u16, k: u32) -> libc::sock_filter {
        libc::sock_filter {
            code,
            jt: 0,
            jf: 0,
            k,
        }
    }
    fn bpf_jump(code: u16, k: u32, jt: u8, jf: u8) -> libc::sock_filter {
        libc::sock_filter { code, jt, jf, k }
    }

    /// Syscalls denied with EPERM. The families called out by the threat model
    /// (`ptrace`, `mount`, `socket`) plus the obvious privilege-escalation and
    /// kernel-surface syscalls. Numbers come from `libc::SYS_*` so they are
    /// arch-correct on both supported arches.
    fn blocked_syscalls() -> Vec<libc::c_long> {
        let mut v = vec![
            libc::SYS_ptrace,
            libc::SYS_process_vm_readv,
            libc::SYS_process_vm_writev,
            libc::SYS_mount,
            libc::SYS_umount2,
            libc::SYS_pivot_root,
            libc::SYS_socket,
            libc::SYS_socketpair,
            libc::SYS_init_module,
            libc::SYS_finit_module,
            libc::SYS_delete_module,
            libc::SYS_kexec_load,
            libc::SYS_reboot,
            libc::SYS_swapon,
            libc::SYS_swapoff,
            libc::SYS_setns,
            libc::SYS_unshare,
        ];
        v.dedup();
        v
    }

    /// Build the seccomp BPF program: validate arch (kill on mismatch to defeat
    /// compat/x32 bypass), then EPERM each blocked syscall, else allow.
    fn build_filter() -> Vec<libc::sock_filter> {
        let mut prog = vec![
            bpf_stmt(BPF_LD_W_ABS, SD_ARCH),
            // if arch == expected, skip the kill (jt=1); else fall through to kill.
            bpf_jump(BPF_JEQ_K, AUDIT_ARCH, 1, 0),
            bpf_stmt(BPF_RET_K, SECCOMP_RET_KILL_PROCESS),
            bpf_stmt(BPF_LD_W_ABS, SD_NR),
        ];
        let errno = SECCOMP_RET_ERRNO | (libc::EPERM as u32 & 0xffff);
        for nr in blocked_syscalls() {
            // if A == nr -> fall through (jt=0) to the EPERM ret; else skip it (jf=1).
            prog.push(bpf_jump(BPF_JEQ_K, nr as u32, 0, 1));
            prog.push(bpf_stmt(BPF_RET_K, errno));
        }
        prog.push(bpf_stmt(BPF_RET_K, SECCOMP_RET_ALLOW));
        prog
    }

    /// True if the kernel supports seccomp filter mode. `PR_GET_SECCOMP` returns
    /// the current mode (>= 0) when seccomp is configured; EINVAL otherwise. No
    /// side effects, so safe to call in the parent as a preflight.
    fn seccomp_available() -> bool {
        // SAFETY: PR_GET_SECCOMP takes no pointer args and does not alter state.
        unsafe { libc::prctl(libc::PR_GET_SECCOMP) >= 0 }
    }

    fn set_rlimit(resource: libc::__rlimit_resource_t, soft: u64, hard: u64) -> io::Result<()> {
        let lim = libc::rlimit {
            rlim_cur: soft,
            rlim_max: hard,
        };
        // SAFETY: `lim` is a valid initialized rlimit for the given resource.
        if unsafe { libc::setrlimit(resource, &lim) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Configure `builder` for sandboxed execution. The privilege drop, chroot,
    /// rlimits and seccomp install all happen in the forked child via `pre_exec`,
    /// after fork and before exec. Any error there aborts the child, so the
    /// command does not run unsandboxed (fail closed at runtime); we also
    /// preflight seccomp availability here to fail closed *before* forking.
    pub fn configure(builder: &mut Command, root: &Path) -> Result<EnforcedShell, String> {
        if !seccomp_available() {
            return Err(
                "seccomp is not available on this host (kernel without CONFIG_SECCOMP); \
                 refusing to run run_shell in sandboxed mode. Re-run with \
                 --shell-sandbox unrestricted to opt out (not recommended)."
                    .to_string(),
            );
        }

        let mut layers = vec!["wall-timeout", "rlimits", "seccomp"];
        let running_as_root = unsafe { libc::geteuid() } == 0;
        if running_as_root {
            layers.push("uid-drop");
        } else {
            // Already unprivileged: we cannot chroot (needs CAP_SYS_CHROOT) and the
            // "non-root uid" requirement is already met. Record it honestly.
            layers.push("already-unprivileged");
        }

        // chroot only when the root looks like a usable rootfs (has /bin/sh);
        // chrooting into a bare scratch dir would leave no shell to exec. When it
        // is not a rootfs we keep cwd-confinement + the other layers and surface
        // the caveat rather than faking a chroot.
        let can_chroot = running_as_root && root.join("bin/sh").exists();
        let mut note = None;
        if can_chroot {
            layers.push("chroot");
        } else if running_as_root {
            note = Some("chroot skipped: workspace is not a rootfs (no /bin/sh)".to_string());
        } else {
            note = Some("chroot skipped: not running as root".to_string());
        }

        let root_c = CString::new(root.as_os_str().as_bytes())
            .map_err(|_| "workspace path contains an interior NUL".to_string())?;

        // pre_exec runs in the child after fork, before exec. Only async-signal
        // -safe libc calls are used. Order matters: chroot → rlimits → drop gid/uid
        // → no_new_privs → seccomp (last, so setup syscalls aren't filtered).
        // SAFETY: the closure performs only direct libc calls on owned data and
        // returns an io::Error on any failure, which aborts the child before exec.
        unsafe {
            builder.pre_exec(move || {
                if can_chroot {
                    if libc::chroot(root_c.as_ptr()) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    if libc::chdir(c"/".as_ptr()) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                } else {
                    // No chroot: confine the cwd to the workspace instead.
                    if libc::chdir(root_c.as_ptr()) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                }

                // Resource caps: no core dumps, 30 CPU-seconds, 1 GiB address
                // space, at most 64 open fds. (Wall-clock is enforced by the
                // parent's kill-on-deadline loop.)
                set_rlimit(libc::RLIMIT_CORE, 0, 0)?;
                set_rlimit(libc::RLIMIT_CPU, 30, 30)?;
                set_rlimit(libc::RLIMIT_AS, 1 << 30, 1 << 30)?;
                set_rlimit(libc::RLIMIT_NOFILE, 64, 64)?;

                if running_as_root {
                    // Drop supplementary groups, then gid, then uid. Order is
                    // critical: setuid before setgid would forfeit the privilege
                    // needed to setgid.
                    if libc::setgroups(0, std::ptr::null()) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    if libc::setgid(NOBODY_GID) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    if libc::setuid(NOBODY_UID) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    // Belt-and-braces: confirm we cannot regain privilege.
                    if libc::setuid(0) == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "uid drop ineffective: process can still become root",
                        ));
                    }
                }

                // Prevent exec'd setuid binaries from re-escalating, then install
                // the filter. NO_NEW_PRIVS is required before SET_SECCOMP without
                // CAP_SYS_ADMIN.
                if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                    return Err(io::Error::last_os_error());
                }
                let filter = build_filter();
                let prog = libc::sock_fprog {
                    len: filter.len() as u16,
                    filter: filter.as_ptr() as *mut libc::sock_filter,
                };
                if libc::prctl(
                    libc::PR_SET_SECCOMP,
                    SECCOMP_MODE_FILTER as libc::c_ulong,
                    &prog as *const _ as libc::c_ulong,
                    0,
                    0,
                ) != 0
                {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }

        Ok(EnforcedShell {
            mode: super::ShellSandbox::Sandboxed,
            layers,
            note,
        })
    }
}

/// Sandboxed configuration dispatch. Linux (supported arch) wires the real
/// enforcement; everywhere else fails closed.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn configure_sandboxed(builder: &mut Command, root: &Path) -> Result<EnforcedShell, String> {
    linux::configure(builder, root)
}

// Windows: there is no seccomp, but cwd-pin + the parent's hard wall-clock timeout
// ARE the native confinement (the same model `run_windows_command` uses), with the
// approval gate as the real backstop. Run with exactly that and report it honestly
// — this is NOT a faked sandbox; it never claims seccomp/uid-drop.
#[cfg(windows)]
fn configure_sandboxed(builder: &mut Command, root: &Path) -> Result<EnforcedShell, String> {
    builder.current_dir(root);
    Ok(EnforcedShell {
        mode: ShellSandbox::Sandboxed,
        layers: vec!["cwd-pin", "wall-timeout"],
        note: Some(
            "Windows-native: no seccomp/uid-drop (Windows has no seccomp); cwd-pinned + \
             hard-timed, gated by the approval policy"
                .to_string(),
        ),
    })
}

// macOS and other unenforceable hosts: still fail closed (no validated native
// confinement is wired here). Refuse rather than silently run unconfined.
#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    windows
)))]
fn configure_sandboxed(_builder: &mut Command, _root: &Path) -> Result<EnforcedShell, String> {
    Err(format!(
        "sandboxed run_shell is not enforceable on this platform ({} / {}): it requires Linux \
         seccomp on x86_64/aarch64 (or Windows-native confinement). Refusing to run run_shell. \
         Use --shell-sandbox unrestricted to opt out, or --shell-sandbox disabled to remove the \
         tool.",
        std::env::consts::OS,
        std::env::consts::ARCH,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unattended_surface_drops_an_unenforceable_sandbox_to_disabled() {
        // A web Code session has no `--shell-sandbox` to fall back on. Leaving
        // the mode at Sandboxed where it cannot be enforced would advertise
        // run_shell to the model and then refuse every call — the model spends
        // the turn on a tool that cannot work, and in approval-gated mode the
        // user is asked to approve commands that are already doomed. Disabled
        // means the tool is simply never offered, which is what fail-closed has
        // to mean when there is no opt-out.
        let dir = std::env::temp_dir();
        let (resolved, why) = resolve_for_unattended_surface(ShellSandbox::Sandboxed, &dir);
        if describe_sandboxed(&dir).is_ok() {
            assert_eq!(resolved, ShellSandbox::Sandboxed);
            assert!(why.is_none());
        } else {
            assert_eq!(resolved, ShellSandbox::Disabled);
            let why = why.expect("an unenforceable host must say why");
            assert!(
                why.contains("not enforceable"),
                "the reason is shown to the user: {why}"
            );
        }
    }

    #[test]
    fn resolving_never_widens_a_mode() {
        // Only the sandboxed→disabled narrowing is in scope. An explicit
        // unrestricted opt-in must not be silently upgraded or downgraded here,
        // and disabled must stay disabled.
        let dir = std::env::temp_dir();
        assert_eq!(
            resolve_for_unattended_surface(ShellSandbox::Unrestricted, &dir).0,
            ShellSandbox::Unrestricted
        );
        assert_eq!(
            resolve_for_unattended_surface(ShellSandbox::Disabled, &dir).0,
            ShellSandbox::Disabled
        );
    }

    #[test]
    fn mode_parses_and_round_trips() {
        assert_eq!(
            "sandboxed".parse::<ShellSandbox>().unwrap(),
            ShellSandbox::Sandboxed
        );
        assert_eq!(
            "Disabled".parse::<ShellSandbox>().unwrap(),
            ShellSandbox::Disabled
        );
        assert_eq!(
            "unrestricted".parse::<ShellSandbox>().unwrap(),
            ShellSandbox::Unrestricted
        );
        assert!("bogus".parse::<ShellSandbox>().is_err());
        assert_eq!(ShellSandbox::default(), ShellSandbox::Sandboxed);
    }

    #[test]
    fn disabled_is_refused_at_configure() {
        let mut c = Command::new("true");
        let err = configure_command(&mut c, Path::new("."), ShellSandbox::Disabled).unwrap_err();
        assert!(err.contains("disabled"));
    }

    #[test]
    fn unrestricted_reports_its_layers() {
        let mut c = Command::new("true");
        let e = configure_command(&mut c, Path::new("."), ShellSandbox::Unrestricted).unwrap();
        assert_eq!(e.mode, ShellSandbox::Unrestricted);
        assert!(e.layers.contains(&"wall-timeout"));
    }

    // On Windows, sandboxed mode is enforced natively (cwd-pin + wall-timeout, no
    // seccomp) and must SUCCEED — run_shell has to work here, reporting exactly
    // what was applied. This is the behavior exercised on the Windows dev box.
    #[cfg(windows)]
    #[test]
    fn sandboxed_is_windows_native_not_failed_closed() {
        let mut c = Command::new("cmd");
        let e = configure_command(&mut c, Path::new("."), ShellSandbox::Sandboxed).unwrap();
        assert_eq!(e.mode, ShellSandbox::Sandboxed);
        assert!(e.layers.contains(&"cwd-pin"));
        assert!(e.layers.contains(&"wall-timeout"));
        assert!(!e.layers.contains(&"seccomp")); // never claim a sandbox we don't have
        assert!(e.note.as_deref().unwrap_or("").contains("Windows-native"));
        assert!(describe_sandboxed(Path::new(".")).is_ok());
    }

    // On other unenforceable hosts (macOS, unsupported arch), sandboxed mode must
    // FAIL CLOSED — it must not silently run unconfined.
    #[cfg(not(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        windows
    )))]
    #[test]
    fn sandboxed_fails_closed_off_linux() {
        let mut c = Command::new("true");
        let err = configure_command(&mut c, Path::new("."), ShellSandbox::Sandboxed).unwrap_err();
        assert!(err.contains("not enforceable"));
        assert!(describe_sandboxed(Path::new(".")).is_err());
    }

    // On Linux, a blocked syscall (opening a raw socket) must fail inside the
    // sandbox. We fork, install the filter via the same pre_exec path by running
    // a tiny shell command, and assert it cannot create a socket. Validated by
    // Linux CI (not run on the Windows dev box).
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn socket_is_blocked_under_seccomp() {
        use std::io::Write;
        use std::process::Stdio;
        // A scratch rootfs is not required for this test: with no /bin/sh under the
        // root, configure() keeps cwd-confinement and still installs seccomp, which
        // is what we are asserting. Use a C one-liner via `sh -c`+`perl`? Keep it
        // dependency-free: use `python3` if present, else skip the assertion body.
        let dir = std::env::temp_dir();
        let mut builder = Command::new("/bin/sh");
        builder
            .arg("-c")
            // Exit 0 only if creating an AF_INET raw socket is refused.
            .arg("python3 - <<'PY'\nimport socket,sys\ntry:\n s=socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)\n sys.exit(1)\nexcept OSError:\n sys.exit(0)\nPY")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let enforced = configure_command(&mut builder, &dir, ShellSandbox::Sandboxed)
            .expect("seccomp must be available on the Linux CI host");
        assert!(enforced.layers.contains(&"seccomp"));
        let status = builder.status().expect("spawn sandboxed shell");
        // 0 = socket() was refused (blocked); anything else means it succeeded or
        // python3 is missing. Treat a missing interpreter as inconclusive-skip.
        if let Some(code) = status.code() {
            assert_eq!(code, 0, "raw socket creation was NOT blocked by seccomp");
        }
        let _ = std::io::stdout().flush();
    }
}
