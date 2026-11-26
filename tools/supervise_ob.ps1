# Keeps the order-book service running (tools\ob_server.py: the order books and trades of the six markets of our
# price, for the bot's order-book models).  Run from the project folder:   powershell -File tools\supervise_ob.ps1
# It needs the research Python (logs\research-venv: numpy, aiohttp, lightgbm, xgboost).  Output: logs\obrec\ob_server.log
# Stop it: create the file logs\obrec\STOP (the service ends, and this loop with it); delete the file before a new start.
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
New-Item -ItemType Directory -Force logs\obrec | Out-Null
while (-not (Test-Path "logs\obrec\STOP")) {
    cmd /c "echo %DATE% %TIME% SUPERVISOR starting the order-book service >> logs\obrec\ob_server.log"
    cmd /c "logs\research-venv\Scripts\python.exe tools\ob_server.py >> logs\obrec\ob_server.log 2>&1"
    cmd /c "echo %DATE% %TIME% SUPERVISOR the service ended >> logs\obrec\ob_server.log"
    Start-Sleep -Seconds 5
}
