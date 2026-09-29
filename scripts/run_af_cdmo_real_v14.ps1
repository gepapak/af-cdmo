param(
    [ValidateSet("development", "frozen_transition", "confirmation")]
    [string]$StudyRole = "development",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "python"
$ReportRoot = Join-Path $Root "official_jao_dual_measure"
$Stem = "af_cdmo_real_${StudyRole}_v14"
$Output = Join-Path $ReportRoot "${Stem}.json"
$Losses = Join-Path $ReportRoot "${Stem}_timestamp_losses.npz"
$Calendar = Join-Path $ReportRoot "${Stem}_calendar_bootstrap.json"
$Manifest = Join-Path $Root "${Stem}_manifest.json"

$Chronologies = @{
    development = @("2025-11-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00")
    frozen_transition = @("2025-09-01T00:00:00+00:00", "2025-09-30T22:00:00+00:00", "2026-01-01T00:00:00+00:00")
    confirmation = @("2026-01-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00", "2026-08-18T00:00:00+00:00")
}
$Dates = $Chronologies[$StudyRole]

Set-Location $Root

Write-Host "[RUN] Compile AF-CDMO real v14 engines"
& $Python -m py_compile `
    scripts\af_cdmo_real_controls_v14.py `
    scripts\benchmark_af_cdmo_real_v14.py `
    scripts\audit_af_cdmo_real_v14.py `
    scripts\analyze_af_cdmo_real_v14_calendar_bootstrap.py
if ($LASTEXITCODE -ne 0) {
    throw "AF-CDMO v14 compilation failed with exit code $LASTEXITCODE"
}

Write-Host "[RUN] Focused AF-CDMO real v14 tests"
& $Python -m pytest `
    tests\test_af_cdmo_real_geometry_v13.py `
    tests\test_af_cdmo_real_extension_v13.py `
    tests\test_benchmark_af_cdmo_real_v13.py `
    tests\test_af_cdmo_real_controls_v14.py `
    tests\test_af_cdmo_real_calendar_bootstrap_v14.py -q
if ($LASTEXITCODE -ne 0) {
    throw "AF-CDMO v14 tests failed with exit code $LASTEXITCODE"
}

$Arguments = @(
    "scripts\benchmark_af_cdmo_real_v14.py",
    "--study_role", $StudyRole,
    "--fit_end", $Dates[0],
    "--validation_end", $Dates[1],
    "--evaluation_end", $Dates[2],
    "--output", $Output,
    "--losses_output", $Losses,
    "--seeds", "7", "42", "123", "2025", "3007", "5001", "8102", "9005", "10001", "11202"
)
if ($Force) {
    $Arguments += "--force"
}

Write-Host "[RUN] AF-CDMO real v14 $StudyRole benchmark"
& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "AF-CDMO real v14 benchmark failed with exit code $LASTEXITCODE"
}

$AuditArguments = @(
    "scripts\audit_af_cdmo_real_v14.py",
    "--report", $Output,
    "--manifest", $Manifest,
    "--required_role", $StudyRole
)
if ($Force) {
    $AuditArguments += "--force"
}

Write-Host "[RUN] Fail-closed AF-CDMO real v14 artifact audit"
& $Python @AuditArguments
if ($LASTEXITCODE -ne 0) {
    throw "AF-CDMO real v14 audit failed with exit code $LASTEXITCODE"
}

$CalendarArguments = @(
    "scripts\analyze_af_cdmo_real_v14_calendar_bootstrap.py",
    "--report", $Output,
    "--output", $Calendar
)
if ($Force) {
    $CalendarArguments += "--force"
}

Write-Host "[RUN] AF-CDMO real v14 calendar-time sensitivity"
& $Python @CalendarArguments
if ($LASTEXITCODE -ne 0) {
    throw "AF-CDMO real v14 calendar sensitivity failed with exit code $LASTEXITCODE"
}

Write-Host "[OK] AF-CDMO real v14 $StudyRole package complete: $Output"
