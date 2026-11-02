# Keeps the bot running: starts main.py --auto --no-repl, and starts it again 15 s after it exits (a crash, a
# "live arm" expiry does NOT exit).  Run from the project folder:   powershell -File tools\supervise.ps1
# Output goes to logs\supervisor.log.  Stop it with Ctrl+C (the bot process ends with it; the browser stays open).
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
New-Item -ItemType Directory -Force logs | Out-Null
$log = Join-Path (Get-Location) "logs\supervisor.log"
while ($true) {
    "$(Get-Date -Format s) SUPERVISOR starting the bot" | Out-File -Append -Encoding ascii $log
    & ".venv\Scripts\python.exe" main.py --auto --no-repl *>> $log
    "$(Get-Date -Format s) SUPERVISOR bot exited with code $LASTEXITCODE, restarting in 15 s" | Out-File -Append -Encoding ascii $log
    Start-Sleep -Seconds 15
}
