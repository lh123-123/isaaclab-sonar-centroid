$ErrorActionPreference = 'Stop'
$packageRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$manifest = Get-ChildItem -LiteralPath $packageRoot -File -Filter '*SHA256.csv' | Select-Object -First 1
if (-not $manifest) { throw 'SHA256 manifest not found' }

$rows = Import-Csv -LiteralPath $manifest.FullName
$bad = @()
foreach ($row in $rows) {
    $relative = $row.RelativePath.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $full = [IO.Path]::GetFullPath((Join-Path $packageRoot $relative))
    if (-not $full.StartsWith($packageRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        $bad += "Escaped root: $relative"
        continue
    }
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        $bad += "Missing: $relative"
        continue
    }
    $file = Get-Item -LiteralPath $full
    $hash = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash
    if ($file.Length -ne [long]$row.Bytes -or $hash -ne $row.SHA256) {
        $bad += "Changed: $relative"
    }
}

$settingsFile = Get-ChildItem -LiteralPath $packageRoot -Filter 'launch_settings.json' -File -Recurse | Select-Object -First 1
if (-not $settingsFile) { throw 'launch_settings.json not found' }
$settings = Get-Content -LiteralPath $settingsFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
$python = Get-Command python -ErrorAction SilentlyContinue
$pythonVersion = if ($python) { (& $python.Source --version 2>&1 | Out-String).Trim() } else { 'not found' }
$pythonCompatible = $false
if ($pythonVersion -match '^Python (\d+)\.(\d+)') {
    $pythonCompatible = ([int]$Matches[1] -gt 3) -or ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -ge 10)
}

[PSCustomObject]@{
    PackageFilesChecked = $rows.Count
    HashMismatches = $bad.Count
    Python = $pythonVersion
    PythonVersionCompatible = $pythonCompatible
    QGCConfiguredPath = $settings.qgc_path
    QGCPathExists = Test-Path -LiteralPath $settings.qgc_path -PathType Leaf
    PiIP = $settings.pi_ip
    WritesToDevices = $false
} | Format-List
if ($bad.Count) { $bad | ForEach-Object { Write-Error $_ }; exit 1 }
