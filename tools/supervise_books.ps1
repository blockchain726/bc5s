# Keeps the recorder of the three extra order books running (tools\record_books.py: Bybit spot, Deribit perpetual,
# Bitfinex spot: the markets bc.game's price follows most and the order-book service does not read).  Only a
# recording: no model reads it yet.  Run from the project folder:   powershell -File tools\supervise_books.ps1
# It needs the research Python (logs\research-venv).  Output: logs\obrec\extra\record_books.log
# Stop it: create the file logs\obrec\extra\STOP (the recorder ends, and this loop with it); delete the file before a
# new start (tools\start_all.ps1 does).
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
New-Item -ItemType Directory -Force logs\obrec\extra | Out-Null
while (-not (Test-Path "logs\obrec\extra\STOP")) {
    cmd /c "echo %DATE% %TIME% SUPERVISOR starting the recorder >> logs\obrec\extra\record_books.log"
    cmd /c "logs\research-venv\Scripts\python.exe tools\record_books.py >> logs\obrec\extra\record_books.log 2>&1"
    cmd /c "echo %DATE% %TIME% SUPERVISOR the recorder ended >> logs\obrec\extra\record_books.log"
    Start-Sleep -Seconds 5
}
