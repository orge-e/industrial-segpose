$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\KeanuReeves\AppData\Local\Programs\Python\Python312\python.exe"
$Environment = Join-Path $ProjectRoot ".venv-qt"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到标准 Python 3.12：$Python"
}
if (-not (Test-Path -LiteralPath $Environment)) {
    & $Python -m venv $Environment
}
& (Join-Path $Environment "Scripts\python.exe") -m pip install -r (Join-Path $ProjectRoot "requirements-qt.txt")
