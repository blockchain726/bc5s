# Keeps the TAP bot alone running (no Up/Down auto mode): starts main.py --no-repl and starts it again 15 s after it
# exits.  Run it in your own terminal so it outlives any assistant session:   powershell -File tools\supervise_tap.ps1
# Output goes to logs\supervisor_tap.log.  Tap live mode and stake come from logs\tap_live_state.json (the page's
# last setting), else .env.  Stop it with Ctrl+C (the browser stays open).
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
New-Item -ItemType Directory -Force logs | Out-Null
$log = Join-Path (Get-Location) "logs\supervisor_tap.log"
while ($true) {
    "$(Get-Date -Format s) SUPERVISOR starting the Tap-only bot" | Out-File -Append -Encoding ascii $log
    & ".venv\Scripts\python.exe" main.py --no-repl *>> $log
    "$(Get-Date -Format s) SUPERVISOR bot exited with code $LASTEXITCODE, restarting in 15 s" | Out-File -Append -Encoding ascii $log
    Start-Sleep -Seconds 15
}
