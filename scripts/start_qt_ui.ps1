$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-qt\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Qt环境尚未建立，请先运行 scripts\setup_qt_ui.ps1"
}
Set-Location -LiteralPath $ProjectRoot
& $Python -m industrial_segpose.ui_qt.app
