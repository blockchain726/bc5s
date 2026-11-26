# Starts everything the Up/Down bot needs, each in its own hidden window, unless it is already running:
#   1. the bot under its supervisor (tools\supervise.ps1: main.py --auto --no-repl, started again 15 s after any exit)
#   2. the order-book service under its supervisor (tools\supervise_ob.ps1): the order-book models and HawkesSign
#      get their answers from it; without it they make no bet
#   3. the price recorder (logs\obrec\record.py, 7 days): the order-book models' fits read its file
#   4. the recorder of three more order books (tools\supervise_books.ps1 -> logs\obrec\extra): nothing reads it yet
#   5. independent paper research and six-hour challenger fits (tools\supervise_model_lab.ps1)
# Use it after a PC restart: double-click start-all.bat in the project folder.  run.bat alone starts the bot WITHOUT
# the automatic mode (no calls, no paper bets) and starts neither of the other two.
# It never switches real orders on.  An arming that was on before the restart is resumed by the bot itself
# (AUTO_RESUME_ARM), as the owner set it.
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
New-Item -ItemType Directory -Force logs\obrec | Out-Null
function Count($names, $pattern) {
    @(Get-CimInstance Win32_Process | Where-Object { $names -contains $_.Name -and $_.CommandLine -like $pattern }).Count
}
$py = @('python.exe', 'python3.13.exe')
$ps = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File')

if (Count $py '*main.py*--auto*') { 'bot: already running' }
elseif (Count $py '*main.py*') {
    'bot: one is running WITHOUT the automatic mode (started with run.bat). Close its window, then run this again.'
}
else {
    Start-Process powershell -ArgumentList ($ps + 'tools\supervise.ps1') -WindowStyle Hidden
    'bot: started (the dashboard answers after about one minute)'
}

if ((Count @('powershell.exe') '*supervise_ob.ps1*') -or (Count $py '*ob_server.py*')) { 'order-book service: already running' }
else {
    Remove-Item logs\obrec\STOP -ErrorAction SilentlyContinue        # the file that ends the service on purpose
    $p = Start-Process powershell -ArgumentList ($ps + 'tools\supervise_ob.ps1') -WindowStyle Hidden -PassThru
    "ob_supervisor $($p.Id) started $(Get-Date -Format o)" | Out-File -Append -Encoding ascii logs\obrec\pids.txt
    'order-book service: started'
}

if (Count $py '*record.py*') { 'price recorder: already running' }
elseif (-not (Test-Path logs\obrec\record.py)) { 'price recorder: logs\obrec\record.py is not on this PC (the order-book fits need it)' }
else {
    $cmd = '".venv\Scripts\python.exe" logs\obrec\record.py logs\obrec 604800 >> logs\obrec\record.out.log 2>> logs\obrec\record.err.log'
    $p = Start-Process cmd -ArgumentList '/c', $cmd -WindowStyle Hidden -PassThru
    "record $($p.Id) started $(Get-Date -Format o)" | Out-File -Append -Encoding ascii logs\obrec\pids.txt
    'price recorder: started (7 days)'
}

# 4. the three order books the service does not read (Bybit spot, Deribit, Bitfinex spot): a recording only
if ((Count @('powershell.exe') '*supervise_books.ps1*') -or (Count $py '*record_books.py*')) { 'extra order books: already recording' }
else {
    Remove-Item logs\obrec\extra\STOP -ErrorAction SilentlyContinue  # the file that ends that recorder on purpose
    $p = Start-Process powershell -ArgumentList ($ps + 'tools\supervise_books.ps1') -WindowStyle Hidden -PassThru
    "books_supervisor $($p.Id) started $(Get-Date -Format o)" | Out-File -Append -Encoding ascii logs\obrec\pids.txt
    'extra order books (Bybit spot, Deribit, Bitfinex spot): recording started'
}

# Independent paper predictions, scoring, and periodic retraining. No real orders.
if ((Count @('powershell.exe') '*supervise_model_lab.ps1*') -or (Count $py '*tools\model_lab.py*')) { 'paper model lab: already running' }
elseif (-not (Test-Path -LiteralPath 'logs\research-venv\Scripts\python.exe')) { 'paper model lab: research Python is not installed' }
else {
    Remove-Item -LiteralPath 'logs\model_lab\STOP' -ErrorAction SilentlyContinue
    Start-Process powershell -ArgumentList ($ps + 'tools\supervise_model_lab.ps1') -WindowStyle Hidden
    'paper model lab: started (logs\model_lab\results.md)'
}
