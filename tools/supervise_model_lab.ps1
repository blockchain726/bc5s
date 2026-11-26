# Independent paper research. Stop with logs\model_lab\STOP. No bot restart or arming.
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$labRoot = Join-Path (Get-Location) 'logs\model_lab'
New-Item -ItemType Directory -Force -Path $labRoot | Out-Null
$labLog = Join-Path $labRoot 'supervisor.log'
$labStop = Join-Path $labRoot 'STOP'
while (-not (Test-Path -LiteralPath $labStop)) {
    "$(Get-Date -Format o) starting the paper model lab" | Out-File -Append -Encoding utf8 $labLog
    & '.\logs\research-venv\Scripts\python.exe' -B tools\model_lab.py *>> $labLog
    if (Test-Path -LiteralPath $labStop) { break }
    "$(Get-Date -Format o) lab exited; retry in 5 seconds" | Out-File -Append -Encoding utf8 $labLog
    Start-Sleep -Seconds 5
}
