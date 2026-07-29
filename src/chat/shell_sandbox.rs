//! OS-level confinement for the `run_shell` tool (Task 1).
//!
//! `run_shell` is the one tool that hands the model a general-purpose execution
//! primitive. The path-confined file tools enforce a canonical-root jail in code
//! (see `tools.rs`), but a shell command can do anything the process can — so it
//! gets a *kernel*-enforced sandbox, not a code check.
//!
//! Three modes (config: `--shell-sandbox`):
//! - [`ShellSandbox::Disabled`] — `run_shell` is not registered at all.
//! - [`ShellSandbox::Sandboxed`] — **the default**. Kernel-enforced confinement
//!   with the workspace as the cwd and the existing hard wall-clock timeout; the
//!   layers underneath are per-platform and listed below.
//! - [`ShellSandbox::Unrestricted`] — explicit opt-in; the command runs
//!   cwd-pinned + timed but otherwise unconfined. Logs a startup warning.
//!
//! **Fail closed, with a real path per platform.** What sandboxed mode applies
//! depends on what the kernel offers, and the answer is reported rather than
//! assumed (see [`EnforcedShell`]):
//!
//! - **Linux** (x86_64/aarch64): seccomp + uid-drop + rlimits + cwd-pin, and
//!   chroot when the root is a usable rootfs. The seccomp filter blocks the
//!   `socket` family, so the shell has no network.
//! - **macOS**: the kernel Sandbox (Seatbelt) through `/usr/bin/sandbox-exec`
//!   with an SBPL profile — writes confined to the workspace plus the process
//!   temp directory, network denied (matching Linux), and credential stores
//!   (`~/.ssh`, keychains, cloud CLI configs) denied for read and write. Reads
//!   elsewhere on the filesystem remain permitted; the workspace read jail is
//!   enforced by the file tools in `tools.rs`, not by this profile.
//! - **Windows** — which has no seccomp — applies the confinement Windows
//!   actually offers (cwd-pin + the hard wall-clock timeout), gated by the
//!   approval policy, and reports **exactly** that (never a faked
//!   seccomp/uid-drop claim); this matches the `run_windows_command` model.
//! - **Anywhere else** (a Linux kernel without `CONFIG_SECCOMP`, an unsupported
//!   arch): sandboxed mode **refuses to run the tool** rather than silently
//!   downgrading.
//!
//! macOS confines by *wrapping* argv rather than by configuring the forked
//! child, which is why [`confined_command`] both builds the command and returns
//! what was enforced: a two-step API would let a caller collect the layer report
//! without the wrapper ever reaching the command line.
//!
//! ## Platform-verification status
//!
//! The **macOS** path is exercised on macOS by `tests::macos_enforcement`, which
//! runs real commands and asserts the kernel refused them (write outside the
//! root, `$HOME`, network, credential reads) — so its reported layers are
//! measured, not claimed. The **Linux** enforcement is compiled behind
//! `cfg(target_os = "linux", target_arch ∈ {x86_64, aarch64})` and is
//! **built-and-gated but not exercised on a Mac or Windows dev box**; it must be
//! validated by Linux CI (the `socket_is_blocked_under_seccomp` test) before the
//! sandbox is trusted in production. See
//! `frontend/design-evidence/DESIGN_LOG.md`.

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

/// Build the confined command that runs `shell_argv` (e.g. `["/bin/sh", "-c",
/// command]`), with `root` as the scratch workspace. Returns the command
/// together with what was actually enforced, or an error that means **refuse to
/// run** (fail closed) — never a silent downgrade. `Disabled` is rejected here
/// too (the tool should not have reached execution).
///
/// This is deliberately the ONLY way to construct a run_shell command. macOS
/// confines by wrapping argv (`sandbox-exec`) rather than by configuring the
/// forked child, so a two-step "build it, then apply confinement" API would let
/// a caller obtain the reported layers without the wrapper ever being applied —
/// the exact shape of "reported but never applied". Handing the argv in means
/// the report and the confinement are produced by one call or not at all.
pub fn confined_command(
    shell_argv: &[&std::ffi::OsStr],
    root: &Path,
    mode: ShellSandbox,
) -> Result<(Command, EnforcedShell), String> {
    let (program, args) = shell_argv
        .split_first()
        .ok_or_else(|| "no shell command to run".to_string())?;

    match mode {
        ShellSandbox::Disabled => Err("run_shell is disabled (shell_sandbox=disabled)".to_string()),
        ShellSandbox::Unrestricted => {
            let mut builder = Command::new(program);
            builder.args(args);
            builder.current_dir(root);
            Ok((builder, EnforcedShell::unrestricted()))
        }
        ShellSandbox::Sandboxed => {
            #[cfg(target_os = "macos")]
            {
                let plan = macos::Plan::build(root)?;
                let mut builder = Command::new(macos::SANDBOX_EXEC);
                // Everything after the prefix is what sandbox-exec execs once the
                // profile is in force.
                builder.args(&plan.argv_prefix()[1..]);
                builder.arg(program).args(args);
                builder.current_dir(root);
                Ok((builder, plan.enforced))
            }
            #[cfg(not(target_os = "macos"))]
            {
                let mut builder = Command::new(program);
                builder.args(args);
                let enforced = configure_sandboxed(&mut builder, root)?;
                Ok((builder, enforced))
            }
        }
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

// ===========================================================================
// macOS enforcement (sandbox-exec / SBPL). Exercised by this platform's own
// tests: `sandbox_exec_confines_writes_to_the_workspace` and friends run real
// commands and assert the kernel refused them, so the layers reported here are
// measured rather than assumed.
// ===========================================================================

#[cfg(target_os = "macos")]
mod macos {
    use super::{EnforcedShell, ShellSandbox};
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};

    /// Apple's sandbox wrapper. An absolute path on purpose: this is the one
    /// program whose identity the confinement depends on, so it is never
    /// resolved through `PATH`.
    pub const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

    /// Credential stores that `run_shell` has no business touching. Denied for
    /// read AND write, after the workspace allow, so the deny wins even when the
    /// workspace happens to sit above one of them. Relative to `$HOME` unless
    /// absolute. This is a macOS-only hardening layer: the Linux profile blocks
    /// syscall families and drops privileges but does not path-deny these.
    const SECRETS: &[&str] = &[
        ".ssh",
        ".aws",
        ".gnupg",
        ".config/gh",
        ".docker",
        ".kube",
        "Library/Keychains",
        "/Library/Keychains",
    ];

    /// A built profile plus the parameters it references and the layers it
    /// actually engages.
    pub struct Plan {
        pub profile: String,
        pub params: Vec<(String, OsString)>,
        pub enforced: EnforcedShell,
    }

    impl Plan {
        pub fn build(root: &Path) -> Result<Self, String> {
            preflight()?;
            // Rules match the kernel's resolved paths, so a workspace reached
            // through a symlink (/tmp -> /private/tmp, /var -> /private/var)
            // must be canonicalised or the jail would allow nothing at all.
            let root = std::fs::canonicalize(root)
                .map_err(|e| format!("cannot resolve the workspace root for the sandbox: {e}"))?;

            let mut layers = vec!["sandbox-exec", "write-jail", "no-network", "cwd-pin"];
            let mut params: Vec<(String, OsString)> =
                vec![("WORKROOT".into(), root.clone().into())];
            let mut rules = String::new();

            // Writes: the workspace, and the per-process temp directory. Temp is
            // not a courtesy — every compiler driver writes there (clang fails
            // with "unable to make temporary file" without it), so a jail that
            // omits it cannot build anything, which is the main thing a coding
            // agent runs.
            rules.push_str("(allow file-write* (subpath (param \"WORKROOT\")))\n");
            match std::fs::canonicalize(std::env::temp_dir()) {
                Ok(tmp) if tmp != root => {
                    params.push(("TMPROOT".into(), tmp.into()));
                    rules.push_str("(allow file-write* (subpath (param \"TMPROOT\")))\n");
                    layers.push("tmp-write");
                }
                // No resolvable temp dir: keep the tighter jail and say so, since
                // build tools will fail rather than escape.
                _ => layers.push("no-tmp-write"),
            }

            let secrets = secret_paths(&root);
            if !secrets.is_empty() {
                rules.push_str("(deny file-read* file-write*");
                for (index, path) in secrets.into_iter().enumerate() {
                    let name = format!("SECRET{index}");
                    rules.push_str(&format!(" (subpath (param \"{name}\"))"));
                    params.push((name, path.into()));
                }
                rules.push_str(")\n");
                layers.push("credential-deny");
            }

            Ok(Plan {
                profile: format!("{BASE_PROFILE}{rules}"),
                params,
                enforced: EnforcedShell {
                    mode: ShellSandbox::Sandboxed,
                    layers,
                    note: Some(
                        "macOS Seatbelt via sandbox-exec: writes confined to the workspace (plus \
                         the process temp dir), network denied, credential stores denied. Reads \
                         elsewhere on the filesystem are still permitted"
                            .to_string(),
                    ),
                },
            })
        }

        /// The argv prefix that applies this plan, ahead of the shell itself.
        pub fn argv_prefix(&self) -> Vec<OsString> {
            let mut argv: Vec<OsString> = vec![SANDBOX_EXEC.into()];
            for (name, value) in &self.params {
                let mut binding = OsString::from(name);
                binding.push("=");
                binding.push(value);
                argv.push("-D".into());
                argv.push(binding);
            }
            argv.push("-p".into());
            argv.push((&self.profile).into());
            argv
        }
    }

    /// Everything denied unless allowed below. `file-read*` stays broad because
    /// build tools read across the whole toolchain prefix; the workspace-only
    /// read jail is enforced by the file TOOLS, not by this profile, and the
    /// note above says so rather than implying a read jail exists here.
    const BASE_PROFILE: &str = "(version 1)\n\
         (deny default)\n\
         (allow process-exec process-fork)\n\
         (allow sysctl-read)\n\
         (allow mach-lookup)\n\
         (allow signal (target self))\n\
         (allow file-read*)\n\
         (allow file-write-data (literal \"/dev/null\") (literal \"/dev/stdout\") \
         (literal \"/dev/stderr\") (literal \"/dev/dtracehelper\"))\n\
         (deny network*)\n";

    fn preflight() -> Result<(), String> {
        let path = Path::new(SANDBOX_EXEC);
        if !path.is_file() {
            return Err(format!(
                "sandboxed run_shell needs {SANDBOX_EXEC}, which is not present on this host. \
                 Use --shell-sandbox unrestricted to opt out, or --shell-sandbox disabled to \
                 remove the tool."
            ));
        }
        Ok(())
    }

    /// Absolute credential paths to deny, skipping any that live inside the
    /// workspace: the agent was pointed at that directory on purpose, and
    /// denying part of its own project would read as a mysterious tool failure.
    fn secret_paths(root: &Path) -> Vec<PathBuf> {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        SECRETS
            .iter()
            .filter_map(|entry| {
                let path = Path::new(entry);
                if path.is_absolute() {
                    Some(path.to_path_buf())
                } else {
                    Some(home.as_ref()?.join(path))
                }
            })
            // Compare on the resolved form when it exists; a path that does not
            // exist yet is still worth denying, in case it appears mid-session.
            .filter(|path| {
                let resolved = std::fs::canonicalize(path);
                let candidate = resolved.as_deref().unwrap_or(path);
                !candidate.starts_with(root)
            })
            .collect()
    }
}

// macOS: confinement comes from the kernel's Sandbox (Seatbelt) via
// `sandbox-exec`, which applies an SBPL profile and then execs the shell. Unlike
// Linux's `pre_exec` route this works by WRAPPING the command, so the profile is
// built here and the argv prefix is applied by `confined_command` — the single
// place that constructs a real run_shell command, so a reported layer cannot
// drift away from an unapplied one.
#[cfg(target_os = "macos")]
fn configure_sandboxed(builder: &mut Command, root: &Path) -> Result<EnforcedShell, String> {
    let plan = macos::Plan::build(root)?;
    builder.current_dir(root);
    Ok(plan.enforced)
}

// Any other host: still fail closed (no validated native confinement is wired
// here). Refuse rather than silently run unconfined.
#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "macos",
    windows
)))]
fn configure_sandboxed(_builder: &mut Command, _root: &Path) -> Result<EnforcedShell, String> {
    Err(format!(
        "sandboxed run_shell is not enforceable on this platform ({} / {}): it requires Linux \
         seccomp on x86_64/aarch64, macOS sandbox-exec, or Windows-native confinement. Refusing \
         to run run_shell. Use --shell-sandbox unrestricted to opt out, or --shell-sandbox \
         disabled to remove the tool.",
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

    /// `["/bin/sh","-c",cmd]` as the borrowed argv `confined_command` takes.
    fn argv(parts: &[&str]) -> Vec<std::ffi::OsString> {
        parts.iter().map(std::ffi::OsString::from).collect()
    }

    fn confine(
        parts: &[&str],
        root: &Path,
        mode: ShellSandbox,
    ) -> Result<(Command, EnforcedShell), String> {
        let owned = argv(parts);
        let borrowed: Vec<&std::ffi::OsStr> = owned.iter().map(|a| a.as_os_str()).collect();
        confined_command(&borrowed, root, mode)
    }

    #[test]
    fn disabled_is_refused_at_configure() {
        let err = confine(&["true"], Path::new("."), ShellSandbox::Disabled).unwrap_err();
        assert!(err.contains("disabled"));
    }

    #[test]
    fn unrestricted_reports_its_layers() {
        let (_, e) = confine(&["true"], Path::new("."), ShellSandbox::Unrestricted).unwrap();
        assert_eq!(e.mode, ShellSandbox::Unrestricted);
        assert!(e.layers.contains(&"wall-timeout"));
    }

    // On Windows, sandboxed mode is enforced natively (cwd-pin + wall-timeout, no
    // seccomp) and must SUCCEED — run_shell has to work here, reporting exactly
    // what was applied. This is the behavior exercised on the Windows dev box.
    #[cfg(windows)]
    #[test]
    fn sandboxed_is_windows_native_not_failed_closed() {
        let (_, e) = confine(&["cmd"], Path::new("."), ShellSandbox::Sandboxed).unwrap();
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
        target_os = "macos",
        windows
    )))]
    #[test]
    fn sandboxed_fails_closed_off_linux() {
        let err = confine(&["true"], Path::new("."), ShellSandbox::Sandboxed).unwrap_err();
        assert!(err.contains("not enforceable"));
        assert!(describe_sandboxed(Path::new(".")).is_err());
    }

    // ---- macOS: real enforcement, measured. Unlike the Linux leg (which this
    // machine cannot exercise), these run actual commands through the actual
    // wrapper and assert the KERNEL refused them. If sandbox-exec ever stops
    // working on a future macOS, these fail rather than quietly reporting a
    // sandbox nobody applied.
    #[cfg(target_os = "macos")]
    mod macos_enforcement {
        use super::super::*;
        use std::path::Path;

        /// Run `command` through the confined builder and return (status_ok, output).
        fn run_confined(root: &Path, command: &str) -> (bool, String) {
            let owned: Vec<std::ffi::OsString> = ["/bin/sh", "-c", command]
                .iter()
                .map(std::ffi::OsString::from)
                .collect();
            let borrowed: Vec<&std::ffi::OsStr> = owned.iter().map(|a| a.as_os_str()).collect();
            let (mut builder, enforced) =
                confined_command(&borrowed, root, ShellSandbox::Sandboxed)
                    .expect("sandbox-exec must be enforceable on macOS");
            assert_eq!(enforced.mode, ShellSandbox::Sandboxed);
            let out = builder.output().expect("spawn the confined shell");
            let mut text = String::from_utf8_lossy(&out.stdout).to_string();
            text.push_str(&String::from_utf8_lossy(&out.stderr));
            (out.status.success(), text)
        }

        #[test]
        fn a_write_inside_the_workspace_is_allowed() {
            let dir = tempfile::tempdir().unwrap();
            let (ok, out) =
                run_confined(dir.path(), "echo confined > inside.txt && cat inside.txt");
            assert!(ok, "an in-workspace write must succeed: {out}");
            assert!(out.contains("confined"), "{out}");
            assert_eq!(
                std::fs::read_to_string(dir.path().join("inside.txt")).unwrap(),
                "confined\n"
            );
        }

        #[test]
        fn a_write_outside_the_workspace_is_denied_by_the_kernel() {
            let dir = tempfile::tempdir().unwrap();
            // Deliberately NOT a `tempfile::tempdir()`: those live under the
            // process temp dir, which is writable on purpose (see
            // `the_temp_tree_is_writable_which_is_the_jails_one_widening`).
            // /private/tmp is a different tree entirely.
            let target = std::path::PathBuf::from(format!(
                "/private/tmp/camelid-sbx-escape-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_file(&target);
            let (ok, out) = run_confined(dir.path(), &format!("echo pwned > {}", target.display()));
            assert!(!ok, "a write outside the root must fail: {out}");
            assert!(
                out.contains("Operation not permitted") || out.contains("not permitted"),
                "the kernel, not our code, must refuse it: {out}"
            );
            assert!(!target.exists(), "nothing may be written outside the root");
        }

        #[test]
        fn the_temp_tree_is_writable_which_is_the_jails_one_widening() {
            // Stated as a test rather than left as a surprise: allowing the
            // process temp dir (required for compilers) means a sibling path in
            // that same tree is writable too. The workspace jail is therefore
            // "workspace + process temp tree", which is what the reported note
            // and the docs say — this pins that claim so it cannot drift.
            let dir = tempfile::tempdir().unwrap();
            let sibling = tempfile::tempdir().unwrap();
            let target = sibling.path().join("in-temp-tree.txt");
            let (ok, out) = run_confined(dir.path(), &format!("echo t > {}", target.display()));
            assert!(
                ok && target.exists(),
                "the temp tree is intentionally writable: {out}"
            );
        }

        #[test]
        fn the_home_directory_is_not_writable() {
            let dir = tempfile::tempdir().unwrap();
            let (ok, out) = run_confined(dir.path(), "echo pwned > \"$HOME/camelid-sbx-escape\"");
            assert!(!ok, "writing into $HOME must fail: {out}");
            if let Some(home) = std::env::var_os("HOME") {
                assert!(!Path::new(&home).join("camelid-sbx-escape").exists());
            }
        }

        #[test]
        fn network_access_is_denied() {
            // Matches the Linux profile, which blocks the socket syscall family
            // outright: run_shell has no egress on either platform.
            let dir = tempfile::tempdir().unwrap();
            let (ok, _) = run_confined(dir.path(), "curl -s -m 5 -o /dev/null https://example.com");
            assert!(!ok, "the shell must not reach the network");
        }

        #[test]
        fn credential_stores_are_not_readable() {
            let dir = tempfile::tempdir().unwrap();
            // Probe with a path that need not exist: a denied read reports
            // "Operation not permitted", a merely-absent one reports "No such
            // file". Only the former proves the deny rule is live.
            let (_, out) = run_confined(dir.path(), "cat \"$HOME/.ssh/id_rsa\" 2>&1");
            assert!(
                out.contains("Operation not permitted"),
                "the ~/.ssh deny rule must be in force (got: {out})"
            );
        }

        #[test]
        fn the_process_temp_dir_stays_writable_so_compilers_work() {
            // clang and friends write intermediates to TMPDIR; a jail without it
            // cannot build anything, which is most of what a coding agent does.
            let dir = tempfile::tempdir().unwrap();
            let (ok, out) = run_confined(
                dir.path(),
                "t=$(mktemp) && echo probe > \"$t\" && cat \"$t\" && rm -f \"$t\"",
            );
            assert!(ok, "the process temp dir must be writable: {out}");
            assert!(out.contains("probe"), "{out}");
        }

        #[test]
        fn the_wrapper_is_actually_applied_and_reported_layers_match() {
            // Guards the one failure mode this seam exists to prevent: layers
            // reported without the wrapper on the command line.
            let dir = tempfile::tempdir().unwrap();
            let owned: Vec<std::ffi::OsString> = ["/bin/sh", "-c", "true"]
                .iter()
                .map(std::ffi::OsString::from)
                .collect();
            let borrowed: Vec<&std::ffi::OsStr> = owned.iter().map(|a| a.as_os_str()).collect();
            let (builder, enforced) =
                confined_command(&borrowed, dir.path(), ShellSandbox::Sandboxed).unwrap();
            assert_eq!(builder.get_program(), "/usr/bin/sandbox-exec");
            let args: Vec<String> = builder
                .get_args()
                .map(|a| a.to_string_lossy().to_string())
                .collect();
            assert!(args.contains(&"-p".to_string()), "{args:?}");
            assert!(
                args.iter().any(|a| a.contains("(deny default)")),
                "the profile must be passed to the wrapper: {args:?}"
            );
            assert!(args.iter().any(|a| a == "/bin/sh"), "{args:?}");
            for layer in ["sandbox-exec", "write-jail", "no-network"] {
                assert!(enforced.layers.contains(&layer), "{:?}", enforced.layers);
            }
            assert!(!enforced.layers.contains(&"seccomp"));
        }

        #[test]
        fn a_workspace_path_with_spaces_and_quotes_is_still_jailed_correctly() {
            // The profile takes the root as a `-D` parameter rather than
            // interpolating it into the SBPL text, so a path containing a quote,
            // a backslash or a paren cannot terminate a string literal and
            // reshape the rules. Both halves matter: the agent's own writes must
            // work, and an outside write must still be refused.
            let parent = tempfile::tempdir().unwrap();
            let root = parent.path().join(r#"we ird"dir (x)\y"#);
            std::fs::create_dir(&root).unwrap();

            let (ok, out) = run_confined(&root, "echo inside > f.txt && cat f.txt");
            assert!(ok, "the agent must be able to write in its own root: {out}");
            assert_eq!(
                std::fs::read_to_string(root.join("f.txt")).unwrap(),
                "inside\n"
            );

            let escape = std::path::PathBuf::from(format!(
                "/private/tmp/camelid-sbx-quote-escape-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_file(&escape);
            let (ok, _) = run_confined(&root, &format!("echo pwned > {}", escape.display()));
            assert!(!ok && !escape.exists(), "the jail must still hold");
        }

        #[test]
        fn a_workspace_reached_through_a_symlink_still_permits_its_own_writes() {
            // /tmp is a symlink to /private/tmp on macOS and the kernel matches
            // resolved paths, so an uncanonicalised root would jail the agent out
            // of its own workspace.
            let dir = tempfile::tempdir().unwrap();
            let link = dir.path().join("link");
            let real = dir.path().join("real");
            std::fs::create_dir(&real).unwrap();
            std::os::unix::fs::symlink(&real, &link).unwrap();
            let (ok, out) = run_confined(&link, "echo via-symlink > f.txt && cat f.txt");
            assert!(ok, "writes through a symlinked root must work: {out}");
            assert_eq!(
                std::fs::read_to_string(real.join("f.txt")).unwrap(),
                "via-symlink\n"
            );
        }
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
