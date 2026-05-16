$WorkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Tools\Anaconda3\envs\sam2\python.exe"

Write-Host "=== Road Pavement AI Diagnosis (WEB MODE) ===" -ForegroundColor Cyan
Write-Host "WorkDir: $WorkDir"
Write-Host "Python:  $Python"

# Public URL of backend port 8000 as seen by remote browsers
# (e.g. VSCode port-forward / devtunnels URL for port 8000)
# If empty, falls back to http://127.0.0.1:8000 — fine for local browser tests.
if (-not $env:BACKEND_URL_PUBLIC) {
    Write-Host ""
    Write-Host "BACKEND_URL_PUBLIC is not set." -ForegroundColor Yellow
    Write-Host "  For VSCode tunnel access, set it BEFORE running this script, e.g.:" -ForegroundColor Yellow
    Write-Host "    `$env:BACKEND_URL_PUBLIC = 'https://<your-id>-8000.<region>.devtunnels.ms'" -ForegroundColor Yellow
    Write-Host "  Continuing with local default http://127.0.0.1:8000 ..." -ForegroundColor Yellow
    $env:BACKEND_URL_PUBLIC = "http://127.0.0.1:8000"
}
Write-Host "BACKEND_URL_PUBLIC: $env:BACKEND_URL_PUBLIC" -ForegroundColor Green

# Check python exists
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: Python not found at $Python" -ForegroundColor Red
    $found = Get-ChildItem "C:\Tools\Anaconda3\envs" -Directory | Where-Object { $_.Name -like "*sam2*" }
    if ($found) {
        $Python = "C:\Tools\Anaconda3\envs\$($found.Name)\python.exe"
        Write-Host "Found: $Python" -ForegroundColor Green
    } else {
        Write-Host "sam2 environment not found!" -ForegroundColor Red
        Read-Host "Press Enter to close"
        exit
    }
}

# Kill existing processes
Write-Host "[1/3] Killing existing processes..." -ForegroundColor Yellow
Get-Process python   -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process uvicorn  -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1
Write-Host "      Done" -ForegroundColor Green

# Check Ollama
Write-Host "[2/3] Checking Ollama..." -ForegroundColor Yellow
try {
    Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null
    Write-Host "      Ollama already running" -ForegroundColor Green
} catch {
    Write-Host "      Starting Ollama..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Minimized
    Start-Sleep -Seconds 3
    Write-Host "      Ollama started" -ForegroundColor Green
}

# Start backend (FastAPI on 0.0.0.0:8000)
Write-Host "[3/3] Starting backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$WorkDir'; & '$Python' main.py"
Write-Host "      Backend window opened" -ForegroundColor Green

Write-Host "      Waiting for backend (max 60s)..."
$count = 0
while ($count -lt 60) {
    Start-Sleep -Seconds 2
    $count += 2
    try {
        Invoke-RestMethod "http://127.0.0.1:8000/" -TimeoutSec 2 | Out-Null
        Write-Host "      Backend ready! ($count s)" -ForegroundColor Green
        break
    } catch {
        Write-Host "      Waiting... ($count s)"
    }
}

# Start Flet in WEB mode on 0.0.0.0:8550
Write-Host "Starting Flet web app on 0.0.0.0:8550..." -ForegroundColor Cyan
if (-not $env:FLET_SECRET_KEY) {
    $env:FLET_SECRET_KEY = -join ((1..48) | ForEach-Object { '{0:x}' -f (Get-Random -Maximum 16) })
    Write-Host "Generated FLET_SECRET_KEY (ephemeral; set your own to persist across restarts)" -ForegroundColor Yellow
}
$childCmd = "cd '$WorkDir'; " +
            "`$env:FLET_WEB='1'; " +
            "`$env:FLET_HOST='0.0.0.0'; " +
            "`$env:FLET_PORT='8550'; " +
            "`$env:BACKEND_URL_PUBLIC='$env:BACKEND_URL_PUBLIC'; " +
            "`$env:FLET_SECRET_KEY='$env:FLET_SECRET_KEY'; " +
            "& '$Python' app.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $childCmd

Write-Host ""
Write-Host "Now in VSCode, forward ports 8000 and 8550 (Ports panel)." -ForegroundColor Cyan
Write-Host "Open the forwarded URL for port 8550 in a browser." -ForegroundColor Cyan
Write-Host "Make sure BACKEND_URL_PUBLIC above matches the forwarded URL of port 8000." -ForegroundColor Cyan
Read-Host "Press Enter to close"
