; Inno Setup script for the Framegrade Windows installer.
;
; Packages the folder build_release.py stages (dist\stage: framegrade.cmd, the
; embeddable python\ and the app\ source). The version comes from the command
; line and has no default, so a build that forgets it fails instead of shipping
; a wrong number:
;
;   ISCC.exe /DAppVersion=0.3.10 packaging\windows\framegrade.iss
;
; build_release.py runs exactly that. The installer is unsigned.
;
; Keep this file pure ASCII. Inno reads a BOM-less script as ANSI.

#ifndef AppVersion
  #error AppVersion is not defined: pass /DAppVersion=x.y.z
#endif

#define AppName "Framegrade"
#define AppURL "https://github.com/Back-Road-Creative/framegrade"
#ifndef StageDir
  #define StageDir "..\..\dist\stage"
#endif

; Everything Framegrade needs that this installer does not contain, said on the
; way out rather than left for the first failed run to discover.
#define GpuNote "Framegrade needs an NVIDIA GPU with a current driver. The blur, grade and render passes run on the GPU; there is no CPU fallback in this install."
#define FFmpegNote "ffmpeg is NOT included. Install it once, then open a new terminal or sign in again:%n%n    winget install -e --id Gyan.FFmpeg"
#define StartNote "Start Framegrade from the Start menu. Your browser opens the console."

[Setup]
; Fixed GUID: how Windows tells an upgrade from a second copy and how the
; uninstaller finds its own entry. Never change.
AppId={{708772F7-8B9D-420A-8ED9-45845AD8D138}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Joe Petrucelli
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
LicenseFile=..\..\LICENSE
; Per-user install: nothing needs administrator rights.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\python\python.exe
OutputDir=..\..\dist
OutputBaseFilename=framegrade-setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "data-folder.txt"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\framegrade.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\python\python.exe"; Flags: runminimized

[Messages]
; Which of the two the wizard shows depends on whether it created any icons,
; so both carry the notes.
FinishedLabel=Setup has installed [name] on your computer.%n%n{#GpuNote}%n%n{#FFmpegNote}%n%n{#StartNote}
FinishedLabelNoIcons=Setup has installed [name] on your computer.%n%n{#GpuNote}%n%n{#FFmpegNote}%n%n{#StartNote}

[Code]
var
  DataPage: TInputDirWizardPage;

{ The launcher reads the chosen folder from this file. An upgrade keeps the
  previous choice as the default. }
function DataFolderFile(): string;
begin
  Result := ExpandConstant('{app}\data-folder.txt');
end;

procedure InitializeWizard();
var
  Previous: AnsiString;
  Initial: string;
begin
  Initial := ExpandConstant('{%USERPROFILE}\Framegrade');
  if LoadStringFromFile(ExpandConstant('{localappdata}\Programs\{#AppName}\data-folder.txt'), Previous) then
    if Trim(String(Previous)) <> '' then
      Initial := Trim(String(Previous));
  DataPage := CreateInputDirPage(wpSelectDir,
    'Data folder',
    'Where should Framegrade keep your footage and results?',
    'Footage, renders, caches and settings go in this folder. Pick a drive with plenty of free space. Uninstalling never deletes it.',
    False, '');
  DataPage.Add('');
  DataPage.Values[0] := Initial;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    ForceDirectories(DataPage.Values[0]);
    SaveStringToFile(DataFolderFile(), DataPage.Values[0], False);
  end;
end;
