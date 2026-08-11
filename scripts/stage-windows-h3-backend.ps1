[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$SdCli,

    [string]$OutDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'camelid-desktop\sidecar'),

    [switch]$SkipRuntimeDlls
)

$ErrorActionPreference = 'Stop'

$source = (Resolve-Path -LiteralPath $SdCli -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "sd-cli is not a file: $source"
}
if ([System.IO.Path]::GetExtension($source) -ine '.exe') {
    throw "the Windows H3 backend must be an .exe: $source"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$destination = Join-Path $OutDir 'sd-cli.exe'
Copy-Item -LiteralPath $source -Destination $destination -Force

$copiedDlls = @()
if (-not $SkipRuntimeDlls) {
    $sourceDirectory = Split-Path -Parent $source
    foreach ($dll in Get-ChildItem -LiteralPath $sourceDirectory -File -Filter '*.dll') {
        $dllDestination = Join-Path $OutDir $dll.Name
        if (Test-Path -LiteralPath $dllDestination -PathType Leaf) {
            $sourceHash = (Get-FileHash -LiteralPath $dll.FullName -Algorithm SHA256).Hash
            $destinationHash = (Get-FileHash -LiteralPath $dllDestination -Algorithm SHA256).Hash
            if ($sourceHash -ne $destinationHash) {
                throw "refusing to replace a different staged DLL named $($dll.Name); use a clean output directory or reconcile the runtime dependency explicitly"
            }
        } else {
            Copy-Item -LiteralPath $dll.FullName -Destination $dllDestination
        }
        $copiedDlls += $dll.Name
    }
}

Write-Host "staged Windows MiniMax-H3 backend: $destination"
if ($copiedDlls.Count -gt 0) {
    Write-Host "staged runtime DLLs: $($copiedDlls -join ', ')"
} elseif (-not $SkipRuntimeDlls) {
    Write-Warning 'no sibling runtime DLLs were found; verify that this sd-cli build is self-contained'
}
