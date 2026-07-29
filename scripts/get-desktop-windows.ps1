# get-desktop-windows.ps1 -- install the prebuilt Camelid Desktop app on
# 64-bit Windows with one command, no toolchain required:
#
#   irm https://raw.githubusercontent.com/timtoole02/Camelid/main/scripts/get-desktop-windows.ps1 | iex
#
# Downloads the signed NSIS installer from the requested release, verifies its
# Authenticode signature, runs a silent per-user install (no admin rights) into
# %LOCALAPPDATA%\Camelid Desktop, and launches the app. Model files and saved
# settings are never touched, so re-running the command updates in place.
#
# A piped invocation (irm | iex) cannot carry script parameters, so
# configuration is environment variables only:
#   CAMELID_DESKTOP_TAG  release tag to install (default: latest)
#   CAMELID_REPO         owner/name to install from (default: timtoole02/Camelid)
#
# Why a signature check instead of a .sha256 sidecar: unlike the macOS DMG,
# which is republished under a stable asset name beside a checksum file, the
# Windows installer keeps its versioned Tauri name and no checksum is
# published for it -- its integrity claim is the Authenticode signature
# (Azure Trusted Signing, RFC 3161 timestamped) applied in the release
# workflow. The same check a user can run by hand is Get-AuthenticodeSignature
# on the downloaded file, and this script fails closed unless Windows reports
# that signature chain as Valid.
#
# The whole script runs inside one script block so the strictness and
# progress preferences it sets do not leak into the calling session, and it
# never calls `exit`, which under `irm | iex` would close the user's console.
& {
  $ErrorActionPreference = 'Stop'
  $ProgressPreference = 'SilentlyContinue' # Invoke-WebRequest is far slower with the progress bar

  if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'Camelid Desktop for Windows requires 64-bit Windows 10 or 11.'
  }

  # Windows PowerShell 5.1 does not offer TLS 1.2 by default on every build,
  # and both api.github.com and the release CDN require it.
  [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

  $repo = if ($env:CAMELID_REPO) { $env:CAMELID_REPO } else { 'timtoole02/Camelid' }
  $tag = if ($env:CAMELID_DESKTOP_TAG) { $env:CAMELID_DESKTOP_TAG } else { 'latest' }

  # The installer asset keeps its versioned name (Camelid.Desktop_<v>_x64-setup.exe),
  # so the release must be resolved through the API rather than a fixed
  # releases/latest/download URL like the macOS script uses.
  $releaseUrl = if ($tag -eq 'latest') {
    "https://api.github.com/repos/$repo/releases/latest"
  } else {
    "https://api.github.com/repos/$repo/releases/tags/$tag"
  }
  try {
    $release = Invoke-RestMethod -Uri $releaseUrl
  } catch {
    throw "could not resolve release '$tag' from $releaseUrl -- $($_.Exception.Message)"
  }

  $asset = @($release.assets | Where-Object { $_.name -like '*x64-setup.exe' })
  if ($asset.Count -ne 1) {
    throw ("release $($release.tag_name) does not publish exactly one Windows desktop installer " +
      "(found $($asset.Count)). The release may predate the desktop app, or the desktop job was " +
      "skipped; download an asset by hand from https://github.com/$repo/releases instead.")
  }
  $asset = $asset[0]

  $installDir = Join-Path $env:LOCALAPPDATA 'Camelid Desktop'
  $workDir = Join-Path $env:TEMP "camelid-desktop-get-$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
  New-Item -ItemType Directory -Path $workDir | Out-Null
  try {
    $setupPath = Join-Path $workDir $asset.name
    Write-Host ("Downloading {0} ({1}, {2:N1} MB) ..." -f $asset.name, $release.tag_name, ($asset.size / 1MB))
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $setupPath

    # Fail closed on anything but a Valid chain: NotSigned, an untrusted root,
    # and a hash mismatch (tampered bytes) all stop here, before anything runs.
    $signature = Get-AuthenticodeSignature -FilePath $setupPath
    if ($signature.Status -ne 'Valid') {
      throw ("the downloaded installer failed Authenticode verification " +
        "($($signature.Status): $($signature.StatusMessage)); refusing to run it")
    }
    Write-Host "Signature OK: $($signature.SignerCertificate.Subject)"

    # Give a running app the chance to shut down its loopback sidecar cleanly
    # before its files are replaced (the desktop process owns the sidecar's
    # shutdown handshake, so only the app window is asked to close). Matching
    # by executable path keeps this scoped to the installed copy -- a
    # standalone `camelid serve` the user runs elsewhere is left alone.
    $installed = {
      @(Get-Process -Name 'camelid-desktop', 'camelid' -ErrorAction SilentlyContinue |
        Where-Object { try { $_.Path -and $_.Path.StartsWith($installDir, [StringComparison]::OrdinalIgnoreCase) } catch { $false } })
    }
    $running = & $installed
    if ($running) {
      Write-Host 'Asking the running Camelid Desktop to quit ...'
      $running | Where-Object { $_.Name -eq 'camelid-desktop' } | ForEach-Object { $null = $_.CloseMainWindow() }
      foreach ($attempt in 1..40) {
        if (-not (& $installed)) { break }
        Start-Sleep -Milliseconds 250
      }
      if (& $installed) {
        throw 'Camelid Desktop did not exit cleanly; close it and run this command again.'
      }
    }

    # /S is the NSIS silent switch; the bundle is configured per-user, so no
    # elevation prompt is involved and -Wait sees the real installer process.
    Write-Host "Installing into $installDir ..."
    $process = Start-Process -FilePath $setupPath -ArgumentList '/S' -Wait -PassThru
    if ($process.ExitCode -ne 0) {
      throw "the installer exited with code $($process.ExitCode)"
    }

    $appExe = Join-Path $installDir 'camelid-desktop.exe'
    if (-not (Test-Path $appExe)) {
      throw "install finished but $appExe is missing"
    }
    Start-Process -FilePath $appExe
    Write-Host "Installed and launched $appExe"
  } finally {
    Remove-Item -Recurse -Force $workDir -ErrorAction SilentlyContinue
  }
}
