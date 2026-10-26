# Proves the BUILT INSTALLER on a clean Windows machine, never the repo.
#
#   windows_proof.ps1 -Installer dist\framegrade-setup-X.Y.Z.exe
#
# Steps: silent per-user install as a standard (non-admin) user into a folder
# whose name has a space and non-ASCII characters; a Microsoft Defender scan of
# the installed folder; the bundled Python starts the console through the real
# launcher and /healthz answers; one CPU-only command (`gomoveshift scan`) on a
# tiny synthetic clip; uninstall, which must leave the data folder alone.
#
# Needs no GPU, no network beyond the machine itself and no credentials, so it
# runs on a GitHub-hosted runner. Run it as administrator (it creates a local
# user). On Actions the shell is pwsh 7.
param([Parameter(Mandatory = $true)][string]$Installer)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false   # a nonzero exit is ours to check
$Installer = (Resolve-Path -LiteralPath $Installer).Path
$started = Get-Date
$os = Get-CimInstance Win32_OperatingSystem
Write-Host "== on $($os.Caption) build $($os.BuildNumber)"

# A hosted runner's Defender service can still be starting when the proof
# begins: the same call is tried again, three attempts 20 s apart.
function Invoke-Defender([scriptblock]$Call, [int]$Tries = 3, [int]$Wait = 20) {
  for ($i = 1; ; $i++) {
    try { return & $Call }
    catch {
      if ($i -ge $Tries) { throw }
      Write-Host "  Defender not ready (attempt $i of $Tries): $($_.Exception.Message)"
      Start-Sleep -Seconds $Wait
    }
  }
}

function New-ProofPassword {
  # Windows wants three character classes: one of each, then 21 more, shuffled.
  $picks = @((48..57 | Get-Random), (65..90 | Get-Random), (97..122 | Get-Random))
  $picks += (48..57 + 65..90 + 97..122) | Get-Random -Count 21
  -join ($picks | Sort-Object { Get-Random } | ForEach-Object { [char]$_ })
}

Write-Host '== 0. a standard user, and a folder it may use'
$user = 'framegradeproof'
$sec = ConvertTo-SecureString (New-ProofPassword) -AsPlainText -Force
New-LocalUser -Name $user -Password $sec -PasswordNeverExpires | Out-Null
Add-LocalGroupMember -Group 'Users' -Member $user
if (Get-LocalGroupMember -Group 'Administrators' | Where-Object { $_.Name -like "*\$user" }) {
  throw "$user is an administrator"
}
$cred = New-Object System.Management.Automation.PSCredential($user, $sec)
$base = 'C:\pröof dir'
$inst = Join-Path $base 'Framegrade ✓'
New-Item -ItemType Directory -Force -Path $base | Out-Null
icacls $base /grant "${user}:(OI)(CI)M" /T | Out-Null
$exe = Join-Path $base (Split-Path $Installer -Leaf)
Copy-Item -LiteralPath $Installer -Destination $exe

# Shows what is left running as the proof user, and the tail of $Trace (a log
# the program writes), so a hung step says what it was waiting on.
function Show-Stuck([string]$Trace) {
  Get-CimInstance Win32_Process | ForEach-Object {
    $o = Invoke-CimMethod -InputObject $_ -MethodName GetOwner -ErrorAction SilentlyContinue
    if ($o -and $o.User -eq $user) { Write-Host "  still running: pid $($_.ProcessId) $($_.CommandLine)" }
  }
  if (-not $Trace) { return }
  if (Test-Path -LiteralPath $Trace) {
    Get-Content -LiteralPath $Trace -Tail 40 | ForEach-Object { Write-Host "  log| $_" }
  } else { Write-Host "  no log at $Trace" }
}

function Stop-UserProcesses {
  Get-CimInstance Win32_Process | ForEach-Object {
    $o = Invoke-CimMethod -InputObject $_ -MethodName GetOwner -ErrorAction SilentlyContinue
    if ($o -and $o.User -eq $user) { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  }
}

# Runs a program as the standard user and returns its exit code. Output goes to
# $Log so it can be shown; a program that must outlive the call is started
# without this function by the caller. The wait is bounded: Start-Process's own wait
# also waits on every descendant, so one stuck child would hang the job until
# its timeout and print nothing. ($ArgLine, never $Args: that is PowerShell's
# automatic variable.)
function Invoke-AsUser([string]$File, [string]$ArgLine, [string]$Log, [string]$Dir = $base,
                       [int]$Minutes = 5, [string]$Trace = '') {
  Write-Host "  run as ${user}: $File $ArgLine"
  $p = Start-Process -FilePath $File -ArgumentList $ArgLine -Credential $cred -LoadUserProfile `
    -WorkingDirectory $Dir -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"
  $null = $p.Handle   # without it ExitCode reads back empty once the process ends
  $done = $p.WaitForExit($Minutes * 60 * 1000)
  Get-Content -LiteralPath $Log, "$Log.err" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  | $_" }
  if (-not $done) {
    Show-Stuck $Trace
    Stop-UserProcesses
    throw "$(Split-Path $File -Leaf) did not finish within $Minutes minutes"
  }
  return $p.ExitCode
}

Write-Host "== 1. silent per-user install into '$inst'"
$code = Invoke-AsUser $exe "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CURRENTUSER /DIR=`"$inst`" /LOG=`"$base\install.log`"" `
  "$base\install.out" -Minutes 20 -Trace "$base\install.log"
if ($code -ne 0) { Get-Content "$base\install.log" -ErrorAction SilentlyContinue | Select-Object -Last 40; throw "installer exited $code" }
$py = Join-Path $inst 'python\python.exe'
foreach ($need in $py, (Join-Path $inst 'framegrade.cmd'), (Join-Path $inst 'app\config.yaml'),
                  (Join-Path $inst 'data-folder.txt'), (Join-Path $inst 'unins000.exe')) {
  if (-not (Test-Path -LiteralPath $need)) { throw "the installer did not write $need" }
}
$dataHome = (Get-Content -LiteralPath (Join-Path $inst 'data-folder.txt') -Raw).Trim()
Write-Host "  data folder: $dataHome"
if (-not (Test-Path -LiteralPath $dataHome)) { throw 'the data folder was not created' }

Write-Host '== 2. Defender scan of the installed folder'
$pref = Invoke-Defender { Get-MpPreference }
foreach ($x in @($pref.ExclusionPath)) { if ($x) { Remove-MpPreference -ExclusionPath $x } }
# Bounded like every other step: a scan that never returns fails with a reason
# instead of being retried.
$scanHung = $false
Invoke-Defender {
  $scan = Start-Job -ScriptBlock { param($Path) Start-MpScan -ScanType CustomScan -ScanPath $Path } -ArgumentList $inst
  if (-not (Wait-Job $scan -Timeout 1200)) { Stop-Job $scan; $script:scanHung = $true; return }
  Receive-Job $scan -ErrorAction Stop
}
if ($scanHung) { throw 'the Defender scan did not finish within 20 minutes' }
$found = @(Get-MpThreatDetection | Where-Object { $_.InitialDetectionTime -ge $started })
if ($found) { throw "Defender flagged the install: $($found.Resources -join ', ')" }
Write-Host '  scanned the installed folder: nothing flagged'

Write-Host '== 3. the console starts and answers /healthz'
# Through the real launcher, as the Start-menu shortcut runs it. It also starts
# the GPU host worker, which cannot run here and is not part of this proof.
Start-Process -FilePath (Join-Path $inst 'framegrade.cmd') -Credential $cred -LoadUserProfile `
  -WorkingDirectory $inst -WindowStyle Hidden | Out-Null
$until = (Get-Date).AddMinutes(3)
$health = $null
while (-not $health) {
  try { $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8787/healthz' -TimeoutSec 5 } catch { }
  if (-not $health) {
    if ((Get-Date) -gt $until) { throw '/healthz did not answer within 3 minutes' }
    Start-Sleep -Seconds 2
  }
}
if ($health.status -ne 'ok') { throw "/healthz said $($health | ConvertTo-Json -Compress)" }
Write-Host '  /healthz: ok'
$owner = (Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.ExecutablePath -eq $py } | Select-Object -First 1)
if (-not $owner) { throw 'no python.exe from the install is running the console' }

function Stop-Install {
  Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($inst) } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Stop-Install

Write-Host '== 4. one CPU-only command on a tiny synthetic clip'
$clip = Join-Path $base 'tiny.avi'
$maker = Join-Path $base 'make_clip.py'
Set-Content -LiteralPath $maker -Encoding ascii -Value @'
import sys
import cv2
import numpy as np
w = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
for i in range(10):
    w.write(np.full((48, 64, 3), i * 20, np.uint8))
w.release()
'@
$code = Invoke-AsUser $py "`"$maker`" `"$clip`"" "$base\clip.out"
if ($code -ne 0 -or -not (Test-Path -LiteralPath $clip)) { throw "could not write the synthetic clip (exit $code)" }
$code = Invoke-AsUser $py "-m gomoveshift scan `"$clip`"" "$base\scan.out" $inst
$out = Get-Content -LiteralPath "$base\scan.out" -Raw
if ($code -ne 0) { throw "gomoveshift scan exited $code" }
foreach ($want in 'Resolution: 64x48', 'Total frames: 10') {
  if ($out -notlike "*$want*") { throw "gomoveshift scan did not print '$want'" }
}

Write-Host '== 5. uninstall leaves the data folder alone'
$marker = Join-Path $dataHome 'proof-marker.txt'
Set-Content -LiteralPath $marker -Value 'mine'
$code = Invoke-AsUser (Join-Path $inst 'unins000.exe') '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART' "$base\uninstall.out" -Minutes 10
if ($code -ne 0) { throw "uninstaller exited $code" }
# The uninstaller re-launches itself from a temp copy, so the call can return first.
$until = (Get-Date).AddSeconds(90)
while ((Test-Path -LiteralPath $py) -and (Get-Date) -lt $until) { Start-Sleep -Seconds 2 }
if (Test-Path -LiteralPath $py) { throw 'the uninstaller left python.exe behind' }
if (-not (Test-Path -LiteralPath $marker)) { throw 'the uninstaller deleted the data folder' }

Remove-LocalUser -Name $user
Write-Host 'PROOF PASSED'
exit 0
