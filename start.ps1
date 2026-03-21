$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Force Python to use UTF-8 for all I/O so Unicode characters (≈, ─, etc.)
# don't crash on the Windows cp1252 console.
$env:PYTHONUTF8 = '1'

if (-not (Test-Path .env)) {
    Write-Host 'ERROR: .env file not found.'
    Write-Host 'Copy .env.example to .env and add your Binance API keys:'
    Write-Host '  Copy-Item .env.example .env'
    exit 1
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host 'ERROR: npm is not installed or not in PATH.'
    Write-Host 'Install Node.js, then run this script again.'
    exit 1
}

$VenvDir = Join-Path $ScriptDir '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPip = Join-Path $VenvDir 'Scripts\pip.exe'

if (-not (Test-Path $VenvPython)) {
    Write-Host 'Creating Python virtual environment...'
    $hasPyLauncher = $null -ne (Get-Command py -ErrorAction SilentlyContinue)
    $hasPython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)

    if (-not $hasPyLauncher -and -not $hasPython) {
        Write-Host 'ERROR: Python is not installed or not in PATH.'
        Write-Host 'Install Python 3, then run this script again.'
        exit 1
    }

    if ($hasPyLauncher) {
        & py -3 -m venv $VenvDir
    } else {
        & python -m venv $VenvDir
    }

    Write-Host 'Installing Python dependencies (first time only)...'
    & $VenvPip install --upgrade pip
    & $VenvPip install -r requirements.txt
    Write-Host 'Python setup complete.'
    Write-Host ''
}

try {
    & $VenvPython -c "import fastapi" 2>$null
} catch {
    Write-Host 'Installing missing Python dependencies...'
    & $VenvPip install -r requirements.txt
}

if (-not (Test-Path 'dashboard/node_modules')) {
    Write-Host 'Installing dashboard dependencies...'
    Push-Location dashboard
    try {
        npm install
    } finally {
        Pop-Location
    }
}

Write-Host ''
Write-Host '=========================================='
Write-Host '      CRYPTO SWING TRADING BOT'
Write-Host '=========================================='
Write-Host '  Backend:   http://localhost:8000'
Write-Host '  API Docs:  http://localhost:8000/docs'
Write-Host '  Dashboard: http://localhost:3000'
Write-Host '=========================================='
Write-Host '  Press Ctrl+C to stop everything'
Write-Host ''

Write-Host '[1/2] Starting backend API server...'

# Use System.Diagnostics.Process with UseShellExecute=false so the child
# inherits this console (like bash's &) and we get a reliable Process object.
$backendPsi = New-Object System.Diagnostics.ProcessStartInfo
$backendPsi.FileName = $VenvPython
$backendPsi.Arguments = "server.py"
$backendPsi.WorkingDirectory = $ScriptDir
$backendPsi.UseShellExecute = $false
$backend = [System.Diagnostics.Process]::Start($backendPsi)

Write-Host '     Waiting for API to be ready...'
$apiReady = $false
for ($i = 1; $i -le 30; $i++) {
    $backend.Refresh()
    if ($backend.HasExited) {
        Write-Host "     ERROR: Backend exited with code $($backend.ExitCode)."
        Write-Host '     Run the following to see the error:'
        Write-Host "       & $VenvPython server.py"
        exit 1
    }
    try {
        $null = Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing -TimeoutSec 2
        Write-Host '     API is ready!'
        $apiReady = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $apiReady -and -not $backend.HasExited) {
    Write-Host '     WARNING: API did not respond within 30s, starting dashboard anyway...'
}

Write-Host '[2/2] Starting dashboard...'

# npm on Windows is npm.cmd — Start-Process tracks the cmd.exe wrapper which
# exits immediately.  Launch via cmd /c so cmd.exe stays alive for the full
# lifetime of the dev server, giving us a reliable process handle.
$frontendPsi = New-Object System.Diagnostics.ProcessStartInfo
$frontendPsi.FileName = "cmd.exe"
$frontendPsi.Arguments = "/c npm run dev"
$frontendPsi.WorkingDirectory = Join-Path $ScriptDir 'dashboard'
$frontendPsi.UseShellExecute = $false
$frontend = [System.Diagnostics.Process]::Start($frontendPsi)

Write-Host ''
Write-Host '  Open http://localhost:3000 in your browser'
Write-Host ''

try {
    while ($true) {
        Start-Sleep -Seconds 1
        $backend.Refresh()
        $frontend.Refresh()
        if ($backend.HasExited -or $frontend.HasExited) { break }
    }
} finally {
    # Kill entire process trees so child node/python processes don't linger
    if (-not $backend.HasExited) {
        taskkill /F /T /PID $backend.Id 2>$null
    }
    if (-not $frontend.HasExited) {
        taskkill /F /T /PID $frontend.Id 2>$null
    }
}
