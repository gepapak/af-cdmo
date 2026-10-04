param(
    [switch]$Smoke,
    [switch]$SkipDownload,
    [switch]$ForceDownload,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "python"
$OutputRoot = Join-Path $Root "external_core_replication"

if ($Smoke) {
    $Start = "2025-01-01"
    $End = "2025-01-15"
    $StartTag = "20250101"
    $EndTag = "20250115"
    $FitEnd = "2025-01-08T00:00:00+00:00"
    $ValidationEnd = "2025-01-11T00:00:00+00:00"
    $EvaluationEnd = "2025-01-15T00:00:00+00:00"
    $Seeds = @("7")
    $Epochs = "8"
    $BatchSize = "64"
    $MaxGroups = "32"
    $MinimumSplitRows = "48"
    $SignAuditDays = "5"
    $Stem = "core_external_smoke_v1"
} else {
    $Start = "2024-10-01"
    $End = "2025-10-01"
    $StartTag = "20241001"
    $EndTag = "20251001"
    $FitEnd = "2025-04-01T00:00:00+00:00"
    $ValidationEnd = "2025-07-01T00:00:00+00:00"
    $EvaluationEnd = "2025-10-01T00:00:00+00:00"
    $Seeds = @("7", "42", "123", "2025", "3007", "5001", "8102", "9005", "10001", "11202")
    $Epochs = "60"
    $BatchSize = "128"
    $MaxGroups = "256"
    $MinimumSplitRows = "500"
    $SignAuditDays = "30"
    $Stem = "core_external_confirmation_v1"
}

$Period = "${StartTag}_${EndTag}"
$Manifest = Join-Path $OutputRoot "core_external_manifest_${Period}.json"
$GeometryAudit = Join-Path $OutputRoot "${Stem}_geometry_audit.json"
$Benchmark = Join-Path $OutputRoot "${Stem}_benchmark.json"
$SummaryRoot = Join-Path $OutputRoot $Stem

Set-Location $Root

Write-Host "================================================================================"
Write-Host "[RUN] Compile Core external-replication engines"
Write-Host "================================================================================"
& $Python -m py_compile `
    scripts\core_external_replication.py `
    scripts\download_jao_core_external_replication.py `
    scripts\audit_af_cdmo_core_geometry.py `
    scripts\benchmark_af_cdmo_core_external.py `
    scripts\summarize_af_cdmo_core_external.py
if ($LASTEXITCODE -ne 0) {
    throw "Core external-replication compilation failed with exit code $LASTEXITCODE"
}

Write-Host "================================================================================"
Write-Host "[RUN] Core external-replication tests"
Write-Host "================================================================================"
& $Python -m pytest tests\test_core_external_replication.py -q
if ($LASTEXITCODE -ne 0) {
    throw "Core external-replication tests failed with exit code $LASTEXITCODE"
}

if (-not $SkipDownload) {
    Write-Host "================================================================================"
    Write-Host "[RUN] Download/freeze JAO Core active constraints and price spreads"
    Write-Host "================================================================================"
    $DownloadArguments = @(
        "-m", "scripts.download_jao_core_external_replication",
        "--start", $Start,
        "--end", $End,
        "--output_root", $OutputRoot,
        "--sign_audit_days", $SignAuditDays
    )
    if ($ForceDownload) {
        $DownloadArguments += "--force_download"
    }
    & $Python @DownloadArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Core external data build failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Core external manifest is missing: $Manifest"
}

Write-Host "================================================================================"
Write-Host "[RUN] Audit conservative Core rank-1 geometry"
Write-Host "================================================================================"
& $Python -m scripts.audit_af_cdmo_core_geometry `
    --manifest $Manifest `
    --output $GeometryAudit `
    --max_groups $MaxGroups
if ($LASTEXITCODE -ne 0) {
    throw "Core geometry audit failed with exit code $LASTEXITCODE"
}

Write-Host "================================================================================"
Write-Host "[RUN] Frozen-method Core external benchmark"
Write-Host "================================================================================"
$BenchmarkArguments = @(
    "-m", "scripts.benchmark_af_cdmo_core_external",
    "--manifest", $Manifest,
    "--geometry_audit", $GeometryAudit,
    "--output", $Benchmark,
    "--fit_end", $FitEnd,
    "--validation_end", $ValidationEnd,
    "--evaluation_end", $EvaluationEnd,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--minimum_split_rows", $MinimumSplitRows,
    "--invariance_seed", "7",
    "--device", $Device,
    "--seeds"
) + $Seeds
if ($Smoke) {
    $BenchmarkArguments += "--smoke_presentations"
}
& $Python @BenchmarkArguments
if ($LASTEXITCODE -ne 0) {
    throw "Core external benchmark failed with exit code $LASTEXITCODE"
}

Write-Host "================================================================================"
Write-Host "[RUN] Summarize Core external evidence"
Write-Host "================================================================================"
& $Python -m scripts.summarize_af_cdmo_core_external `
    --report $Benchmark `
    --output_dir $SummaryRoot
if ($LASTEXITCODE -ne 0) {
    throw "Core external summary failed with exit code $LASTEXITCODE"
}

Write-Host "[OK] Core external replication complete: $Benchmark"
