param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "restart", "status")]
    [string]$Command = "status",

    [int]$Port = 8999,
    [string]$HostName = "127.0.0.1"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeDir = Join-Path $ProjectRoot ".runtime"
$PidFile = Join-Path $RuntimeDir "llm-view.pid"
$OutLog = Join-Path $RuntimeDir "server.out.log"
$ErrLog = Join-Path $RuntimeDir "server.err.log"

function Ensure-RuntimeDir {
    if (-not (Test-Path -LiteralPath $RuntimeDir)) {
        New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
    }
}

function Get-TrackedPid {
    if (-not (Test-Path -LiteralPath $PidFile)) {
        return $null
    }

    $raw = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    if (-not $raw) {
        return $null
    }

    return [int]$raw
}

function Get-TrackedProcess {
    $pidValue = Get-TrackedPid
    if ($null -eq $pidValue) {
        return $null
    }

    return Get-Process -Id $pidValue -ErrorAction SilentlyContinue
}

function Get-PortListener {
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Test-IsOurServer {
    param([int]$ProcessId)

    $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue).CommandLine
    if (-not $commandLine) {
        return $false
    }

    return ($commandLine -match "llm[-_ ]?view") -or ($commandLine -like "*$ProjectRoot*")
}

function Wait-ForServer {
    param([System.Diagnostics.Process]$Process)

    for ($i = 0; $i -lt 40; $i++) {
        if ($Process.HasExited) {
            return $false
        }

        if (Get-PortListener) {
            return $true
        }

        Start-Sleep -Milliseconds 250
    }

    return $false
}

function Start-App {
    Ensure-RuntimeDir

    $trackedProcess = Get-TrackedProcess
    if ($null -ne $trackedProcess) {
        Write-Output "LLM View is already running. PID: $($trackedProcess.Id)"
        Write-Output "URL: http://$HostName`:$Port"
        return
    }

    if (Test-Path -LiteralPath $PidFile) {
        Remove-Item -LiteralPath $PidFile
    }

    $listener = Get-PortListener
    if ($null -ne $listener) {
        $ownerPid = $listener.OwningProcess
        if (Test-IsOurServer -ProcessId $ownerPid) {
            Write-Output "Found an untracked LLM View server on port $Port (PID $ownerPid). Replacing it."
            Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 600
        }
        else {
            Write-Error "Port $Port is already in use by PID $ownerPid. Stop that process or choose another port."
        }
    }

    $env:LLM_VIEW_HOST = $HostName
    $env:LLM_VIEW_PORT = [string]$Port

    $process = Start-Process `
        -FilePath "uv" `
        -ArgumentList @("run", "llm-view") `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -WindowStyle Hidden `
        -PassThru

    Set-Content -LiteralPath $PidFile -Value $process.Id

    if (Wait-ForServer -Process $process) {
        $listener = Get-PortListener
        $serverPid = if ($null -ne $listener) { $listener.OwningProcess } else { $process.Id }
        Set-Content -LiteralPath $PidFile -Value $serverPid

        Write-Output "LLM View started. PID: $serverPid"
        Write-Output "URL: http://$HostName`:$Port"
        Write-Output "Logs: $OutLog"
        return
    }

    Write-Error "LLM View did not start cleanly. Check $ErrLog"
}

function Stop-App {
    $trackedProcess = Get-TrackedProcess

    if ($null -eq $trackedProcess) {
        if (Test-Path -LiteralPath $PidFile) {
            Remove-Item -LiteralPath $PidFile
        }

        $listener = Get-PortListener
        if ($null -ne $listener) {
            $ownerPid = $listener.OwningProcess
            if (Test-IsOurServer -ProcessId $ownerPid) {
                Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
                Write-Output "Stopped untracked LLM View server on port $Port (PID $ownerPid)."
                return
            }

            Write-Output "No tracked PID found. Port $Port is currently used by PID $ownerPid (not LLM View)."
            return
        }

        Write-Output "LLM View is not running."
        return
    }

    Stop-Process -Id $trackedProcess.Id
    [void]$trackedProcess.WaitForExit(5000)

    if (Test-Path -LiteralPath $PidFile) {
        Remove-Item -LiteralPath $PidFile
    }

    Write-Output "LLM View stopped. PID: $($trackedProcess.Id)"
}

function Show-Status {
    $trackedProcess = Get-TrackedProcess
    $listener = Get-PortListener

    if ($null -ne $trackedProcess) {
        Write-Output "LLM View tracked process is running. PID: $($trackedProcess.Id)"
    }
    else {
        Write-Output "LLM View has no tracked running process."
    }

    if ($null -ne $listener) {
        Write-Output "Port $Port is listening. PID: $($listener.OwningProcess)"
        Write-Output "URL: http://$HostName`:$Port"
    }
    else {
        Write-Output "Port $Port is not listening."
    }
}

switch ($Command) {
    "start" { Start-App }
    "stop" { Stop-App }
    "restart" {
        Stop-App
        Start-App
    }
    "status" { Show-Status }
}
