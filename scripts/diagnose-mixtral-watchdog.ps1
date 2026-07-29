[CmdletBinding()]
param(
    [string]$ApiBase = "http://127.0.0.1:8181",
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [string]$Prompt = "Hello",
    [ValidateRange(10, 8192)]
    [int]$MaxTokens = 50,
    [ValidateRange(0, 8191)]
    [int]$DiagnosticGeneratedIndex = 9,
    [ValidateRange(30, 86400)]
    [int]$TimeoutSeconds = 1800,
    [ValidateRange(1, 60)]
    [int]$PollSeconds = 5,
    [string]$ApiKey,
    [string]$ApiKeyFile,
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
if ($ApiKey -and $ApiKeyFile) {
    throw "Use either -ApiKey or -ApiKeyFile, not both."
}
if ($ApiKeyFile) {
    $ApiKey = (Get-Content -LiteralPath $ApiKeyFile -Raw).Trim()
}
if ($DiagnosticGeneratedIndex -ge $MaxTokens) {
    throw "-DiagnosticGeneratedIndex must be smaller than -MaxTokens."
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $OutputDirectory) {
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $OutputDirectory = Join-Path $repoRoot "qa\local-artifacts\mixtral-watchdog-$stamp"
}
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
if (-not $outputRoot.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDirectory must stay inside the Camelid repository: $repoRoot"
}
[System.IO.Directory]::CreateDirectory($outputRoot) | Out-Null

$base = $ApiBase.TrimEnd("/")
$headers = [System.Net.Http.Headers.HttpRequestHeaders]
$client = [System.Net.Http.HttpClient]::new()
$probeClient = [System.Net.Http.HttpClient]::new()
$probeClient.Timeout = [TimeSpan]::FromSeconds([Math]::Max(2, $PollSeconds))
if ($ApiKey) {
    $client.DefaultRequestHeaders.Authorization =
        [System.Net.Http.Headers.AuthenticationHeaderValue]::new("Bearer", $ApiKey)
    $probeClient.DefaultRequestHeaders.Authorization =
        [System.Net.Http.Headers.AuthenticationHeaderValue]::new("Bearer", $ApiKey)
}

$body = @{
    model = $Model
    messages = @(@{ role = "user"; content = $Prompt })
    max_tokens = $MaxTokens
    temperature = 0
    stream = $false
    camelid_dense_diagnostic_generated_index = $DiagnosticGeneratedIndex
    camelid_logit_token_ids = @(1691, 1047)
} | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText(
    (Join-Path $outputRoot "request.json"),
    $body,
    [System.Text.UTF8Encoding]::new($false)
)

$request = [System.Net.Http.HttpRequestMessage]::new(
    [System.Net.Http.HttpMethod]::Post,
    "$base/v1/chat/completions"
)
$request.Content = [System.Net.Http.StringContent]::new(
    $body,
    [System.Text.Encoding]::UTF8,
    "application/json"
)
$cancel = [System.Threading.CancellationTokenSource]::new()
$sendTask = $client.SendAsync(
    $request,
    [System.Net.Http.HttpCompletionOption]::ResponseContentRead,
    $cancel.Token
)
$started = [DateTimeOffset]::UtcNow
$deadline = $started.AddSeconds($TimeoutSeconds)
$snapshotPath = Join-Path $outputRoot "watchdog.jsonl"

try {
    while (-not $sendTask.IsCompleted -and [DateTimeOffset]::UtcNow -lt $deadline) {
        $snapshot = [ordered]@{
            observed_utc = [DateTime]::UtcNow.ToString("o")
            elapsed_seconds = [Math]::Round(
                ([DateTimeOffset]::UtcNow - $started).TotalSeconds,
                3
            )
        }
        foreach ($probe in @(
            @{ Name = "health"; Path = "/v1/health" },
            @{ Name = "slots"; Path = "/slots" }
        )) {
            try {
                $text = $probeClient.GetStringAsync("$base$($probe.Path)").GetAwaiter().GetResult()
                $snapshot[$probe.Name] = $text | ConvertFrom-Json
            }
            catch {
                $snapshot["$($probe.Name)_error"] = $_.Exception.Message
            }
        }
        try {
            $metrics = $probeClient.GetStringAsync("$base/metrics").GetAwaiter().GetResult()
            [System.IO.File]::WriteAllText(
                (Join-Path $outputRoot "metrics-latest.prom"),
                $metrics,
                [System.Text.UTF8Encoding]::new($false)
            )
        }
        catch {
            $snapshot["metrics_error"] = $_.Exception.Message
        }
        [System.IO.File]::AppendAllText(
            $snapshotPath,
            (($snapshot | ConvertTo-Json -Compress -Depth 12) + [Environment]::NewLine),
            [System.Text.UTF8Encoding]::new($false)
        )
        Start-Sleep -Seconds $PollSeconds
    }

    if (-not $sendTask.IsCompleted) {
        $cancel.Cancel()
        $timeout = [ordered]@{
            status = "external_watchdog_timeout"
            timeout_seconds = $TimeoutSeconds
            completed_utc = [DateTime]::UtcNow.ToString("o")
            interpretation = "Inspect engine_active_generated_tokens and engine_stalled_seconds: token progress indicates slow MoE compute; a flat token count with a growing stall gauge identifies an in-flight forward that made no token-boundary progress."
        }
        [System.IO.File]::WriteAllText(
            (Join-Path $outputRoot "result.json"),
            ($timeout | ConvertTo-Json -Depth 6),
            [System.Text.UTF8Encoding]::new($false)
        )
        Write-Error "Mixtral request exceeded the external watchdog. Evidence: $outputRoot"
        exit 2
    }

    $response = $sendTask.GetAwaiter().GetResult()
    $responseBody = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    [System.IO.File]::WriteAllText(
        (Join-Path $outputRoot "response.json"),
        $responseBody,
        [System.Text.UTF8Encoding]::new($false)
    )
    $result = [ordered]@{
        status = if ($response.IsSuccessStatusCode) { "completed" } else { "http_error" }
        http_status = [int]$response.StatusCode
        elapsed_seconds = [Math]::Round(
            ([DateTimeOffset]::UtcNow - $started).TotalSeconds,
            3
        )
        completed_utc = [DateTime]::UtcNow.ToString("o")
        evidence_directory = $outputRoot
    }
    [System.IO.File]::WriteAllText(
        (Join-Path $outputRoot "result.json"),
        ($result | ConvertTo-Json -Depth 6),
        [System.Text.UTF8Encoding]::new($false)
    )
    $result | ConvertTo-Json -Depth 6
    if (-not $response.IsSuccessStatusCode) {
        exit 1
    }
}
finally {
    $cancel.Dispose()
    $request.Dispose()
    $client.Dispose()
    $probeClient.Dispose()
}
