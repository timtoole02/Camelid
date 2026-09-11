#Requires -Version 7
# Windows lifetime smoke for Camelid Desktop (P7, docs/fabric-backends/AGENT.md).
#
# Drives the real debug camelid-desktop.exe: its Tauri window, tray, job object and
# single-instance IPC. Only the engine is a stand-in: camelid-desktop/examples/fake_sidecar.rs
# staged as target\debug\camelid.exe, which advertises no optional flags, so the crash check
# isolates the job object. Prints one "LIFETIME-SMOKE PASS: <name>" line per check and exits
# non-zero with a named message on the first failure.
#
# The window is found by EnumWindows on the desktop PID, the exact title and visibility,
# never Process.MainWindowHandle: debug builds are console-subsystem, so the "main window"
# can be the console, and WM_CLOSE to a console is a Ctrl+C exit that proves nothing.
# Arguments always go in arrays, and no parameter is named $args (AGENT.md section 7).

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $PSScriptRoot
$desktopExe = Join-Path $repo 'target\debug\camelid-desktop.exe'
$engineExe = Join-Path $repo 'target\debug\camelid.exe'
foreach ($required in @($desktopExe, $engineExe)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "missing $required" }
}
$logDir = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$appData = Join-Path $env:APPDATA 'app.camelid.desktop'
$preferencePath = Join-Path $appData 'desktop-lifetime.json'
$savedPreference = if (Test-Path -LiteralPath $preferencePath) {
    [IO.File]::ReadAllBytes($preferencePath)
} else { $null }
Remove-Item Env:FAKE_SIDECAR_ADVERTISE -ErrorAction SilentlyContinue

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class LifetimeSmokeWin32 {
    public delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr hwnd, StringBuilder text, int max);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);

    public static IntPtr[] FindWindows(uint pid, string title, bool visibleOnly) {
        var found = new List<IntPtr>();
        EnumWindows((hwnd, lParam) => {
            uint owner;
            GetWindowThreadProcessId(hwnd, out owner);
            if (owner != pid) { return true; }
            var text = new StringBuilder(512);
            GetWindowText(hwnd, text, text.Capacity);
            if (text.ToString() == title && (!visibleOnly || IsWindowVisible(hwnd))) { found.Add(hwnd); }
            return true;
        }, IntPtr.Zero);
        return found.ToArray();
    }
}
'@

$WM_CLOSE = 0x0010

function Write-Pass([string]$Name) { Write-Output "LIFETIME-SMOKE PASS: $Name" }

function Wait-Until([scriptblock]$Condition, [int]$Seconds, [string]$What) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Condition) { return }
        Start-Sleep -Milliseconds 200
    }
    throw "timed out after $Seconds s waiting for: $What"
}

function Test-Alive([int]$ProcessId) {
    $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Get-EngineProcesses {
    @(Get-CimInstance Win32_Process -Filter "Name = 'camelid.exe'" |
        Where-Object { $_.ExecutablePath -eq $engineExe })
}

function Get-HealthStatus([int]$Port) {
    try {
        (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri "http://127.0.0.1:$Port/v1/health").StatusCode
    } catch { 0 }
}

function Set-LifetimePreference([bool]$KeepRunning) {
    New-Item -ItemType Directory -Force -Path $appData | Out-Null
    # notice_shown: the one-time notice is a modal dialog the smoke cannot dismiss.
    $json = @{ version = 1; keep_engine_running_when_window_closes = $KeepRunning; notice_shown = $true } |
        ConvertTo-Json -Compress
    [IO.File]::WriteAllText($preferencePath, $json)
}

function Start-Desktop([string]$Tag) {
    $log = Join-Path $logDir "camelid-desktop-$Tag.stderr.log"
    $process = Start-Process -FilePath $desktopExe -NoNewWindow -PassThru -RedirectStandardError $log
    # Caching the handle now is what makes ExitCode readable after the process exits.
    $null = $process.Handle
    $script:found = @()
    Wait-Until {
        $script:found = [LifetimeSmokeWin32]::FindWindows([uint32]$process.Id, 'Camelid Desktop', $true)
        $script:found.Length -gt 0
    } 90 "a visible 'Camelid Desktop' window owned by pid $($process.Id) ($Tag)"
    $hwnd = $script:found[0]
    Wait-Until { (Get-EngineProcesses).Count -eq 1 } 60 "exactly one engine process ($Tag)"
    $engineId = [int](Get-EngineProcesses)[0].ProcessId
    $script:listen = $null
    Wait-Until {
        $script:listen = Get-NetTCPConnection -OwningProcess $engineId -State Listen -ErrorAction SilentlyContinue
        $null -ne $script:listen
    } 60 "the engine listening ($Tag)"
    $port = [int]@($script:listen)[0].LocalPort
    Wait-Until { (Get-HealthStatus $port) -eq 200 } 60 "/v1/health 200 on 127.0.0.1:$port ($Tag)"
    [pscustomobject]@{ Process = $process; Hwnd = $hwnd; EngineId = $engineId; Port = $port; Log = $log }
}

function Stop-Leftovers {
    Get-Process -Name 'camelid-desktop' -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $desktopExe } | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-EngineProcesses | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

$failed = $false
try {
    Stop-Leftovers

    # 1. Background mode off (the default): closing the window quits and stops the engine.
    Set-LifetimePreference $false
    $off = Start-Desktop 'off'
    [void][LifetimeSmokeWin32]::PostMessage($off.Hwnd, $WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero)
    if (-not $off.Process.WaitForExit(15000)) { throw 'LIFETIME-OFF: the desktop did not exit after WM_CLOSE' }
    $code = $off.Process.ExitCode
    if ($code -ne 0) { throw ('LIFETIME-OFF: desktop exit code 0x{0:X8}, expected 0' -f $code) }
    Wait-Until { -not (Test-Alive $off.EngineId) } 10 "LIFETIME-OFF: engine pid $($off.EngineId) gone"
    Write-Pass 'LIFETIME-OFF close quits'

    # 2. Background mode on: closing hides the window; the same engine keeps serving.
    Set-LifetimePreference $true
    $on = Start-Desktop 'on'
    [void][LifetimeSmokeWin32]::PostMessage($on.Hwnd, $WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero)
    Wait-Until { -not [LifetimeSmokeWin32]::IsWindowVisible($on.Hwnd) } 10 'LIFETIME-ON: the window hidden'
    Start-Sleep -Seconds 2
    if ($on.Process.HasExited) {
        throw "LIFETIME-ON: the desktop exited on close (exit code $($on.Process.ExitCode)); see $($on.Log)"
    }
    if (-not (Test-Alive $on.EngineId)) { throw "LIFETIME-ON: engine pid $($on.EngineId) is gone" }
    if ((Get-HealthStatus $on.Port) -ne 200) { throw "LIFETIME-ON: /v1/health on $($on.Port) is not 200" }
    Write-Pass 'LIFETIME-ON close hides'

    # 3. A second launch hands over to the running instance: no second engine.
    $secondLog = Join-Path $logDir 'camelid-desktop-second.stderr.log'
    $second = Start-Process -FilePath $desktopExe -NoNewWindow -PassThru -RedirectStandardError $secondLog
    $null = $second.Handle
    if (-not $second.WaitForExit(15000)) {
        Stop-Process -Id $second.Id -Force -ErrorAction SilentlyContinue
        throw 'SINGLE-INSTANCE: the second launch did not exit'
    }
    Wait-Until { [LifetimeSmokeWin32]::IsWindowVisible($on.Hwnd) } 10 "SINGLE-INSTANCE: the first instance's window shown"
    $engines = (Get-EngineProcesses).Count
    if ($engines -ne 1) { throw "SINGLE-INSTANCE: $engines engine processes, expected 1" }
    if (-not (Test-Alive $on.EngineId)) { throw 'SINGLE-INSTANCE: the original engine is gone' }
    Write-Pass 'SINGLE-INSTANCE'

    # 4. The desktop dies without running any of its code: only the job object can act.
    Stop-Process -Id $on.Process.Id -Force
    Wait-Until { -not (Test-Alive $on.EngineId) } 5 "LIFETIME-ON crash: engine pid $($on.EngineId) killed with its job"
    Write-Pass 'LIFETIME-ON crash'
} catch {
    $failed = $true
    Write-Output "LIFETIME-SMOKE FAIL: $($_.Exception.Message)"
    Get-ChildItem -Path $logDir -Filter 'camelid-desktop-*.stderr.log' -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Output "--- $($_.Name)"; Get-Content -LiteralPath $_.FullName -Tail 40 }
} finally {
    Stop-Leftovers
    if ($null -ne $savedPreference) {
        [IO.File]::WriteAllBytes($preferencePath, $savedPreference)
    } else {
        Remove-Item -LiteralPath $preferencePath -ErrorAction SilentlyContinue
    }
}
if ($failed) { exit 1 }
exit 0
