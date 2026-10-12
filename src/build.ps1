param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (Test-Path -LiteralPath $Python -PathType Leaf) {
    $pythonExecutable = (Resolve-Path -LiteralPath $Python).Path
} else {
    $pythonExecutable = (Get-Command $Python -CommandType Application -ErrorAction Stop).Source
}
Push-Location $PSScriptRoot
try {
    & $pythonExecutable -m pip install -r (Join-Path $projectRoot 'requirements.txt') pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "安装依赖失败" }
    & $pythonExecutable package_app.py
    if ($LASTEXITCODE -ne 0) { throw "打包失败" }
    $programDirectory = Join-Path $projectRoot 'app'
    New-Item -ItemType Directory -Path $programDirectory -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'dist\SudokuAssistant.exe') -Destination $programDirectory
    Write-Host "程序已生成：app\SudokuAssistant.exe"
} finally {
    Pop-Location
}
