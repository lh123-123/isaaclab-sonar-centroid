$ErrorActionPreference = 'Stop'
$sourceFile = Join-Path $PSScriptRoot 'QGC_ROV_FlyView_Actions.json'
$targetFolder = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'QGroundControl\MavlinkActions'
New-Item -ItemType Directory -Force -Path $targetFolder | Out-Null
$targetFile = Join-Path $targetFolder 'QGC_ROV_FlyView_Actions.json'
Copy-Item -LiteralPath $sourceFile -Destination $targetFile -Force
Write-Host "已安装QGC动作文件：$targetFile"
Write-Host '请重启QGC，然后在 Application Settings -> MAVLink Actions 中选择该文件。'
