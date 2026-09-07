param(
    [Parameter(Mandatory = $true)]
    [string]$ImagePath,

    [string]$OutputDirectory = "",

    [string]$Label = "workpiece",

    [switch]$SkipPrelabel
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = "D:\Anaconda\python.exe"
$labelmeExe = "D:\Tools\Labelme\venv\Scripts\labelme.exe"

if (-not (Test-Path -LiteralPath $ImagePath)) {
    throw "找不到图像：$ImagePath"
}

if (-not (Test-Path -LiteralPath $projectPython)) {
    throw "找不到项目 Python：$projectPython"
}

if (-not (Test-Path -LiteralPath $labelmeExe)) {
    throw "找不到 Labelme：$labelmeExe"
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $projectRoot "datasets\annotation_workbench"
}

$absoluteImage = (Resolve-Path -LiteralPath $ImagePath).Path
$absoluteOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
$annotationPath = Join-Path $absoluteOutput (([System.IO.Path]::GetFileNameWithoutExtension($absoluteImage)) + ".json")

if (-not $SkipPrelabel) {
    & $projectPython (Join-Path $projectRoot "scripts\prepare_labelme_annotation.py") `
        prelabel `
        --image $absoluteImage `
        --output $absoluteOutput `
        --label $Label

    if ($LASTEXITCODE -ne 0) {
        throw "自动预标注失败，退出代码：$LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $annotationPath)) {
    throw "找不到标注文件：$annotationPath"
}

Write-Host "正在打开 Labelme：$annotationPath" -ForegroundColor Cyan
& $labelmeExe $annotationPath
