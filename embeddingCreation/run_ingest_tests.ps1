# Run unit tests for ingest pipeline (probes, bucketing, dynamic batching) and batch_builder.
# Usage: from repo root or this folder:
#   powershell -ExecutionPolicy Bypass -File .\embeddingCreation\run_ingest_tests.ps1
# Or cd embeddingCreation first, then: .\run_ingest_tests.ps1

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    $py = Get-Command python -ErrorAction SilentlyContinue
}
if (-not $py) {
    Write-Error "Python not found (try py launcher or python on PATH)."
}

& $py.Source -3 -m unittest test_ingest_pipeline.py test_batch_builder.py -v
