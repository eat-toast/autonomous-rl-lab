param(
    [Parameter(Position = 0)]
    [ValidateSet('test', 'train', 'evaluate')]
    [string]$Command = 'test',
    [string]$Config = 'configs/smoke.json',
    [string]$Checkpoint,
    [switch]$FinalTest,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Path $PSScriptRoot -Parent)
if (-not $PythonPath) {
    $localPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
        $PythonPath = (Resolve-Path -LiteralPath '.venv\Scripts\python.exe').Path
    } elseif (Test-Path -LiteralPath $localPython) {
        $PythonPath = $localPython
    } else {
        $PythonPath = 'python'
    }
}
switch ($Command) {
    'test' { & $PythonPath -m unittest discover -s tests -v }
    'train' { & $PythonPath -m rl_lab train --config $Config }
    'evaluate' {
        if (-not $Checkpoint) { throw '-Checkpoint is required' }
        if ($FinalTest) { & $PythonPath -m rl_lab evaluate --checkpoint $Checkpoint --final-test }
        else { & $PythonPath -m rl_lab evaluate --checkpoint $Checkpoint }
    }
}
exit $LASTEXITCODE
