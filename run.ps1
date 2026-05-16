$WorkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Tools\Anaconda3\envs\sam2\python.exe"

Write-Host "=== Road Pavement AI Diagnosis ===" -ForegroundColor Cyan
Write-Host "WorkDir: $WorkDir"
Write-Host "Python: $Python"

# Check python exists
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: Python not found at $Python" -ForegroundColor Red
    Write-Host "Trying to find sam2 python..." -ForegroundColor Yellow
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
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process uvicorn -ErrorAction SilentlyContinue | Stop-Process -Force
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

# Start backend
Write-Host "[3/3] Starting backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$WorkDir'; & '$Python' main.py"
Write-Host "      Backend window opened" -ForegroundColor Green

# Wait for backend
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

# Start app
Write-Host "Starting app..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$WorkDir'; & '$Python' app.py"

Write-Host "All done! You can close this window." -ForegroundColor Green
Read-Host "Press Enter to close"
