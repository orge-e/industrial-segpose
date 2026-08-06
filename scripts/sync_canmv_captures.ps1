param(
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$DeviceName = "CanMV"
)

$ErrorActionPreference = "Stop"
$shell = New-Object -ComObject Shell.Application
$computer = $shell.Namespace(17)
$device = $computer.Items() | Where-Object { $_.Name -eq $DeviceName } | Select-Object -First 1
if ($null -eq $device) { throw "未检测到CanMV便携设备" }

function Child-Folder($parent, [string]$name) {
    $namespace = $shell.Namespace($parent)
    if ($null -eq $namespace) { return $null }
    return $namespace.Items() | Where-Object { $_.IsFolder -and $_.Name -eq $name } | Select-Object -First 1
}

$sdcard = Child-Folder $device "sdcard"
if ($null -eq $sdcard) { throw "CanMV中未找到sdcard目录" }
$application = Child-Folder $sdcard "industrial_vision"
if ($null -eq $application) { throw "sdcard中未找到industrial_vision目录" }
$captures = Child-Folder $application "captures"
if ($null -eq $captures) { throw "industrial_vision中未找到captures目录" }

$destinationPath = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
$destinationNamespace = $shell.Namespace($destinationPath)
if ($null -eq $destinationNamespace) { throw "无法打开本地同步目录" }
$destinationNamespace.CopyHere($captures, 1044)

$capturePath = Join-Path $destinationPath "captures"
$previousCount = -1
$stableRounds = 0
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    Start-Sleep -Milliseconds 500
    $count = @(Get-ChildItem -LiteralPath $capturePath -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|tif|tiff)$' }).Count
    if ($count -gt 0 -and $count -eq $previousCount) { $stableRounds++ } else { $stableRounds = 0 }
    if ($stableRounds -ge 3) { Write-Output $capturePath; exit 0 }
    $previousCount = $count
}
throw "同步超时：未能从CanMV完整复制采集图像"
