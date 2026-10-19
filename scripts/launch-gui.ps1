param([switch]$Dashboard)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repoRoot
$logPath = Join-Path $repoRoot 'setup.log'

try {
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    $uvPath = if ($uvCommand) { $uvCommand.Source } else { Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
    if (-not (Test-Path -LiteralPath $uvPath)) {
        throw 'uv is not installed. Install uv using the link in README.md, then double-click the launcher again.'
    }
    # Sync each launch so a GitHub Desktop pull is picked up automatically.
    $env:UV_NO_PROGRESS = '1'
    $setup = Start-Process -FilePath $uvPath -ArgumentList 'sync', '--locked', '--no-dev' -WorkingDirectory $repoRoot -RedirectStandardError $logPath -WindowStyle Hidden -Wait -PassThru
    if ($setup.ExitCode -ne 0) {
        throw "Dependency setup failed. See $logPath for details."
    }
    $pythonPath = Join-Path $repoRoot '.venv\Scripts\pythonw.exe'
    $moduleName = if ($Dashboard) { 'vibration_analysis.dashboard' } else { 'vibration_analysis.gui' }
    Start-Process -FilePath $pythonPath -ArgumentList '-m', $moduleName -WorkingDirectory $repoRoot -WindowStyle Hidden
} catch {
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'Vibration analysis setup', 'OK', 'Error') | Out-Null
}
