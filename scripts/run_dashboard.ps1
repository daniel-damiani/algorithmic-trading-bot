# Start the LAN trading dashboard (binds 0.0.0.0:8000)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Error "Virtual environment not found. Run: py -3.11 -m venv .venv"
    exit 1
}

$Port = 8000
try {
    $cfg = Get-Content "config\config.yaml" -Raw
    if ($cfg -match 'port:\s*(\d+)') {
        $Port = [int]$Matches[1]
    }
} catch { }

# Free port if another Python process is already listening (common after dev restarts)
$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $listeners) {
    $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
    if ($proc -and $proc.ProcessName -eq "python") {
        Write-Host "Stopping stale Python on port ${Port} (PID $($proc.Id))..."
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    } elseif ($proc) {
        Write-Warning "Port ${Port} is in use by $($proc.ProcessName) (PID $($proc.Id)). Stop it or change monitoring.dashboard.port in config.yaml"
        exit 1
    }
}

$LanIp = (
    Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.PrefixOrigin -ne 'WellKnown' } |
    Select-Object -First 1 -ExpandProperty IPAddress
)

Write-Host ""
Write-Host "Dashboard URLs (do NOT use 0.0.0.0 in the browser):"
Write-Host "  This PC:    http://localhost:${Port}"
if ($LanIp) {
    Write-Host "  LAN/phone:  http://${LanIp}:${Port}"
}
Write-Host ""

.\.venv\Scripts\python.exe -m src.api.server
