//! A Windows kill-on-close Job Object for process-tree teardown.
//!
//! Assign a freshly spawned child to the job; terminating the job (or dropping
//! its last handle) kills the child AND every descendant it spawned. Used by
//! `run_windows_command` so a timeout cannot leave orphaned PowerShell — and,
//! in a later phase, subagent — processes running. `std::process::Child::kill`
//! alone only reaps the direct child, not its tree.

use std::os::windows::io::{AsRawHandle, RawHandle};
use std::process::Child;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32,
};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};

/// An owned job-object handle. Created with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`,
/// so closing it (Drop) terminates any process still in the job.
pub struct JobObject {
    handle: HANDLE,
}

// SAFETY: a Windows job-object handle is a process-wide kernel handle, not tied
// to the thread that created it; create/assign/terminate/close are all valid from
// any thread. This lets a JobObject be held in the process-global subagent
// registry (a `Mutex`-guarded static). It is never used concurrently without that
// lock.
unsafe impl Send for JobObject {}

impl JobObject {
    /// Create a new unnamed kill-on-close job object.
    pub fn new() -> std::io::Result<Self> {
        // SAFETY: null security attributes + null name create a fresh unnamed job
        // and return a null handle on failure.
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let job = JobObject { handle };

        // SAFETY: the struct is plain-old-data (integers/handles); all-zero is a
        // valid initial state, and we set the one field we rely on below.
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: `info` is a correctly sized, zero-initialized struct of the type
        // the ExtendedLimitInformation class expects; we pass its true byte length.
        let ok = unsafe {
            SetInformationJobObject(
                job.handle,
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(info) as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(job)
    }

    /// Assign a spawned child process to the job. Descendants the child spawns
    /// after assignment are captured by the job too.
    pub fn assign(&self, process: RawHandle) -> std::io::Result<()> {
        // SAFETY: `process` is a live process handle owned by the caller's
        // `std::process::Child` for the duration of this call.
        let ok = unsafe { AssignProcessToJobObject(self.handle, process as HANDLE) };
        if ok == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// Create the mandatory kill-on-close guard for a child launched with
    /// `CREATE_SUSPENDED`, assign it, and only then resume its sole initial
    /// thread. The child cannot execute user code (and therefore cannot launch
    /// an escaping descendant) before assignment. Every create, assign, or
    /// resume failure terminates and reaps it while failing closed.
    pub fn contain_suspended(child: &mut Child) -> Result<Self, String> {
        let job = super::tools::establish_child_guard(
            child,
            "Windows Job Object",
            Self::new,
            |job, child| job.assign(child.as_raw_handle()),
            kill_and_reap,
        )?;
        if let Err(error) = resume_initial_thread(child.id()) {
            job.terminate();
            let cleanup_error = kill_and_reap(child).err();
            let mut message = format!("could not resume contained child: {error}");
            if let Some(cleanup_error) = cleanup_error {
                message.push_str(&format!("; child cleanup also failed: {cleanup_error}"));
            }
            return Err(message);
        }
        Ok(job)
    }

    /// Immediately terminate every process in the job.
    pub fn terminate(&self) {
        // SAFETY: terminating a valid, owned job handle; the exit code is arbitrary.
        unsafe {
            TerminateJobObject(self.handle, 1);
        }
    }
}

fn kill_and_reap(child: &mut Child) -> Result<(), String> {
    // kill may report InvalidInput when the child raced to exit; wait is still
    // mandatory because it reaps either state.
    let _ = child.kill();
    child
        .wait()
        .map(|_| ())
        .map_err(|error| format!("could not reap spawned child: {error}"))
}

/// Resume the one initial thread of a process created with CREATE_SUSPENDED.
/// Rust's `Child` owns only the process handle, so recover that thread through a
/// bounded system snapshot. More than one thread is an invariant violation: do
/// not guess which one is primary or resume a potentially injected thread.
fn resume_initial_thread(process_id: u32) -> std::io::Result<()> {
    // SAFETY: no pointer arguments. INVALID_HANDLE_VALUE signals failure.
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snapshot == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
        return Err(std::io::Error::last_os_error());
    }
    struct Snapshot(HANDLE);
    impl Drop for Snapshot {
        fn drop(&mut self) {
            // SAFETY: the handle came from CreateToolhelp32Snapshot and is
            // closed exactly once here.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
    let snapshot = Snapshot(snapshot);

    // SAFETY: THREADENTRY32 is plain-old-data; Windows requires dwSize to be
    // initialized before the first enumeration call.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
    let mut found = None;
    // SAFETY: snapshot is live and entry points to a correctly sized struct.
    let mut available = unsafe { Thread32First(snapshot.0, &mut entry) } != 0;
    while available {
        if entry.th32OwnerProcessID == process_id && found.replace(entry.th32ThreadID).is_some() {
            return Err(std::io::Error::other(
                "suspended child had more than one initial thread",
            ));
        }
        // SAFETY: same live snapshot and output struct as Thread32First.
        available = unsafe { Thread32Next(snapshot.0, &mut entry) } != 0;
    }
    let thread_id = found.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "suspended child's initial thread was not found",
        )
    })?;

    // SAFETY: request only resume access to the enumerated thread; the returned
    // null handle indicates failure.
    let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, thread_id) };
    if thread.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    struct ThreadHandle(HANDLE);
    impl Drop for ThreadHandle {
        fn drop(&mut self) {
            // SAFETY: the handle came from OpenThread and is closed once here.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
    let thread = ThreadHandle(thread);
    // SAFETY: thread is a live handle with THREAD_SUSPEND_RESUME access.
    let previous_count = unsafe { ResumeThread(thread.0) };
    if previous_count == u32::MAX {
        return Err(std::io::Error::last_os_error());
    }
    if previous_count != 1 {
        return Err(std::io::Error::other(format!(
            "initial thread had unexpected suspend count {previous_count}"
        )));
    }
    Ok(())
}

impl Drop for JobObject {
    fn drop(&mut self) {
        // SAFETY: `handle` came from CreateJobObjectW and has not been closed.
        // Closing the last handle on a kill-on-close job terminates survivors.
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::windows::process::CommandExt;
    use windows_sys::Win32::System::Threading::CREATE_SUSPENDED;

    #[test]
    fn suspended_child_runs_only_after_job_assignment_and_resume() {
        let command = std::env::var_os("ComSpec").expect("Windows ComSpec");
        let mut child = std::process::Command::new(command)
            .args(["/D", "/C", "exit 7"])
            .creation_flags(CREATE_SUSPENDED)
            .spawn()
            .unwrap();
        assert!(child.try_wait().unwrap().is_none());

        let _job = JobObject::contain_suspended(&mut child).unwrap();
        assert_eq!(child.wait().unwrap().code(), Some(7));
    }
}
