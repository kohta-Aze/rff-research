param([switch]$Standalone)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$sharedPython = Join-Path $repoRoot '2.define-baseline-model/00_common/data_preprocessing/venv/Scripts/python.exe'
$localPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $localPython)) {
    if (-not (Test-Path -LiteralPath $sharedPython)) {
        throw 'Shared Python is unavailable. Create .venv with Python 3.13 and install requirements.txt.'
    }
    & $sharedPython -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed.' }
}
if ($Standalone) {
    $sharedLink = Join-Path $PSScriptRoot '.venv/Lib/site-packages/rff_shared_packages.pth'
    if (Test-Path -LiteralPath $sharedLink) { Remove-Item -LiteralPath $sharedLink }
    & $localPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. libmr requires a compatible compiler or wheel.' }
} else {
    # Reuse already-verified native packages; never install into the existing environment.
    $sharedSite = Join-Path (Split-Path -Parent (Split-Path -Parent $sharedPython)) 'Lib/site-packages'
    if (-not (Test-Path -LiteralPath $sharedSite)) { throw 'Shared site-packages is unavailable.' }
    $linkPath = Join-Path $PSScriptRoot '.venv/Lib/site-packages/rff_shared_packages.pth'
    [System.IO.File]::WriteAllText($linkPath, $sharedSite + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}
& $localPython (Join-Path $PSScriptRoot 'scripts/check_environment.py')
if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed.' }
