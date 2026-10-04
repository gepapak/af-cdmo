param(
    [string]$Python = "python",
    [string]$Report = "af_cdmo_probabilistic_headroom/quotient_tangent_risk_confirmation_v5.json",
    [string]$SummaryDir = "af_cdmo_probabilistic_headroom/qct_risk_confirmation_v5_summary",
    [string]$TemporalAuditDir = "af_cdmo_probabilistic_headroom/qct_risk_temporal_robustness_v5",
    [int]$BootstrapReplicates = 4000
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Invoke-Checked {
    param(
        [string]$Label,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host ("=" * 80)
    Write-Host "[RUN] $Label"
    Write-Host "$Python $($Arguments -join ' ')"
    Write-Host ("=" * 80)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$ReportPath = Join-Path $Root $Report
if (-not (Test-Path -LiteralPath $ReportPath)) {
    throw "Frozen confirmation report is not available: $ReportPath"
}

$Status = & $Python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(d.get('status',''))" $ReportPath
if ($LASTEXITCODE -ne 0 -or $Status.Trim() -ne "confirmation_complete") {
    throw "Frozen confirmation report is not complete: $ReportPath"
}

Invoke-Checked "Compile QCT-Risk confirmation utilities" @(
    "-m", "py_compile",
    "af_cdmo_probabilistic_headroom/confirm_quotient_tangent_risk_v5.py",
    "af_cdmo_probabilistic_headroom/summarize_qct_risk_confirmation_v5.py",
    "af_cdmo_probabilistic_headroom/audit_qct_risk_temporal_robustness_v5.py"
)

Invoke-Checked "Run focused QCT-Risk tests" @(
    "-m", "pytest", "-q", "tests/test_qct_risk_confirmation_v5.py"
)

Invoke-Checked "Generate confirmation tables and decision summary" @(
    "af_cdmo_probabilistic_headroom/summarize_qct_risk_confirmation_v5.py",
    "--report", $Report,
    "--output_dir", $SummaryDir
)

Invoke-Checked "Run post-confirmation temporal robustness audit" @(
    "af_cdmo_probabilistic_headroom/audit_qct_risk_temporal_robustness_v5.py",
    "--report", $Report,
    "--output_dir", $TemporalAuditDir,
    "--block_days", "1", "7", "14", "28",
    "--replicates", $BootstrapReplicates.ToString()
)

$ManifestPath = Join-Path $Root "af_cdmo_probabilistic_headroom/qct_risk_post_confirmation_v5_manifest.json"
$ManifestScript = @'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
paths = [pathlib.Path(value).resolve() for value in sys.argv[2:]]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = []
for path in paths:
    if path.is_dir():
        candidates = sorted(item for item in path.rglob("*") if item.is_file())
    else:
        candidates = [path]
    for candidate in candidates:
        files.append({
            "path": candidate.relative_to(root).as_posix(),
            "sha256": sha256(candidate),
            "bytes": candidate.stat().st_size,
        })
print(json.dumps({"status": "complete", "files": files}, indent=2))
'@

$ManifestJson = $ManifestScript | & $Python - $Root $ReportPath (Join-Path $Root $SummaryDir) (Join-Path $Root $TemporalAuditDir)
if ($LASTEXITCODE -ne 0) {
    throw "Post-confirmation manifest generation failed"
}
$ManifestJson | Set-Content -LiteralPath $ManifestPath -Encoding utf8

Write-Host ""
Write-Host "[OK] QCT-Risk post-confirmation pipeline complete"
Write-Host "Report: $ReportPath"
Write-Host "Summary: $(Join-Path $Root $SummaryDir)"
Write-Host "Temporal audit: $(Join-Path $Root $TemporalAuditDir)"
Write-Host "Manifest: $ManifestPath"
