#requires -version 5
<#
.SYNOPSIS
    Stage the NVRTC redistributable DLLs next to camelid.exe so a downloaded
    Windows release runs on the GPU with only the NVIDIA *driver* installed
    (no CUDA Toolkit required on the end-user's machine).

.DESCRIPTION
    camelid links cudarc with `fallback-dynamic-loading`, so it loads the CUDA
    driver (nvcuda.dll, shipped with the GPU driver) and the NVRTC runtime
    compiler dynamically at startup. The resident decode engine compiles its
    kernels at runtime via NVRTC -- and NVRTC ships with the CUDA *Toolkit*, not
    the driver. Without it, the GPU is detected but the kernel compile fails and
    inference silently falls back to the CPU.

    This script copies the NVRTC redistributables from an installed CUDA Toolkit
    into the release directory next to camelid.exe. At launch,
    `ensure_cuda_runtime_on_path()` (src/main.rs) puts the exe's own directory on
    PATH first, so the shipped pair is found before any system toolkit.

    The CUDA *driver* (nvcuda.dll) is NOT redistributable and is NOT copied -- it
    is provided by the user's NVIDIA GPU driver, which any GeForce/RTX user has.

.PARAMETER OutDir
    Directory to stage the DLLs into (the folder that holds camelid.exe).
    Defaults to <repo>/dist/Camelid.

.PARAMETER CudaBin
    CUDA Toolkit `bin` directory to copy from. Auto-detected from CUDA_PATH* and
    the standard install root when omitted.

.EXAMPLE
    pwsh scripts/package-windows-cuda.ps1
    pwsh scripts/package-windows-cuda.ps1 -OutDir C:\release\Camelid
#>
param(
    [string]$OutDir,
    [string]$CudaBin
)

$ErrorActionPreference = 'Stop'

# Repo root is the parent of this script's scripts/ directory.
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $repoRoot 'dist\Camelid' }

# --- Locate the CUDA Toolkit bin directory ------------------------------------
function Find-CudaBin {
    # 1. Explicit CUDA_PATH / CUDA_PATH_V* environment variables.
    $envPaths = Get-ChildItem Env: |
        Where-Object { $_.Name -eq 'CUDA_PATH' -or $_.Name -like 'CUDA_PATH_V*' } |
        ForEach-Object { Join-Path $_.Value 'bin' } |
        Where-Object { Test-Path $_ }
    if ($envPaths) { return ($envPaths | Select-Object -First 1) }

    # 2. Standard install root: pick the newest vXX.Y\bin.
    $root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $root) {
        $newest = Get-ChildItem $root -Directory |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName 'bin' } |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1
        if ($newest) { return $newest }
    }
    return $null
}

if (-not $CudaBin) { $CudaBin = Find-CudaBin }
if (-not $CudaBin -or -not (Test-Path $CudaBin)) {
    Write-Error "Could not locate a CUDA Toolkit 'bin' directory. Install the CUDA Toolkit (>= 12.x) or pass -CudaBin <path>."
}
Write-Host "CUDA bin:  $CudaBin"

# --- Resolve the NVRTC redistributable DLLs -----------------------------------
# nvrtc64_*.dll      -> the runtime compiler (e.g. nvrtc64_120_0.dll)
# nvrtc-builtins64_* -> its required builtins (e.g. nvrtc-builtins64_129.dll)
#
# `.alt.` variants are an alternate JIT backend cudarc never asks for; skipping
# them keeps ~86 MB out of every download. Mirrors the same skip in
# scripts/package-linux-cuda.sh -- without it the staged set trips the
# `.alt.dll` forbidden rule in scripts/check-release-artifact.mjs.
$patterns = @('nvrtc64_*.dll', 'nvrtc-builtins64_*.dll')
$dlls = foreach ($p in $patterns) {
    $found = Get-ChildItem -Path (Join-Path $CudaBin $p) -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike '*.alt.dll' }
    if (-not $found) {
        Write-Warning "No files matched '$p' in $CudaBin"
    }
    $found
}
$dlls = $dlls | Where-Object { $_ } | Sort-Object FullName -Unique
if (-not $dlls) {
    Write-Error "Found no NVRTC DLLs in $CudaBin -- nothing to package."
}

# --- Stage into the release directory -----------------------------------------
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}
$OutDir = (Resolve-Path $OutDir).Path
Write-Host "Staging to: $OutDir`n"

foreach ($dll in $dlls) {
    Copy-Item -LiteralPath $dll.FullName -Destination $OutDir -Force
    $mb = '{0:N1}' -f ($dll.Length / 1MB)
    Write-Host ("  + {0,-32} {1,6} MB" -f $dll.Name, $mb)
}

# --- Sanity check: warn if the exe is missing or the core pair is incomplete ---
$exe = Join-Path $OutDir 'camelid.exe'
if (-not (Test-Path $exe)) {
    Write-Warning "camelid.exe not found in $OutDir -- build it (cargo build --release --bin camelid) and place it here so the shipped NVRTC is loaded from the exe directory."
}
$haveNvrtc    = Get-ChildItem (Join-Path $OutDir 'nvrtc64_*.dll') -ErrorAction SilentlyContinue
$haveBuiltins = Get-ChildItem (Join-Path $OutDir 'nvrtc-builtins64_*.dll') -ErrorAction SilentlyContinue
if (-not $haveNvrtc -or -not $haveBuiltins) {
    Write-Warning "NVRTC pair incomplete in $OutDir (need both nvrtc64_*.dll and nvrtc-builtins64_*.dll) -- the GPU path will fall back to CPU."
} else {
    Write-Host "`nDone. The release in $OutDir is self-contained: a user with only the NVIDIA driver gets GPU acceleration."
}
