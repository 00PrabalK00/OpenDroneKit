; OpenDroneKit Windows installer (Inno Setup 6).
;
; Do not compile this by hand: tools/installer/build.py stages the payload and passes
; every value below with /D. Per-user install by default (no administrator rights);
; the user can still choose "install for all users" on the first page, which elevates.

#ifndef AppVersion
  #error AppVersion must be defined by tools/installer/build.py
#endif
#ifndef StageDir
  #error StageDir must be defined by tools/installer/build.py
#endif

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}{#SignLabel}
AppPublisher={#Publisher}
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}
VersionInfoVersion={#AppVersion}
VersionInfoDescription={#AppName} installer{#SignLabel}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBase}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\OpenDroneKit.ico
UninstallDisplayName={#AppName} {#AppVersion}
LicenseFile={#LicenseFile}
#if Unsigned == "1"
InfoBeforeFile={#UnsignedNotice}
#endif
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
CloseApplications=yes
#if SignEnabled == "1"
SignTool=odksign
SignedUninstaller=yes
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}\{#AppName}"; Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launch.py"""; WorkingDir: "{app}\odk"; IconFilename: "{app}\OpenDroneKit.ico"
Name: "{autoprograms}\{#AppName}\Download AI models"; Filename: "{app}\runtime\python.exe"; Parameters: """{app}\launch.py"" --download-models --all --pause"; WorkingDir: "{app}\odk"; IconFilename: "{app}\OpenDroneKit.ico"
Name: "{autoprograms}\{#AppName}\Self-test"; Filename: "{app}\runtime\python.exe"; Parameters: """{app}\launch.py"" --self-test --pause"; WorkingDir: "{app}\odk"; IconFilename: "{app}\OpenDroneKit.ico"
Name: "{autoprograms}\{#AppName}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launch.py"""; WorkingDir: "{app}\odk"; IconFilename: "{app}\OpenDroneKit.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\runtime\python.exe"; Parameters: """{app}\launch.py"" --download-models --all --pause"; WorkingDir: "{app}\odk"; Description: "Check for downloadable AI model weights (first run)"; Flags: postinstall skipifsilent unchecked
Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\launch.py"""; WorkingDir: "{app}\odk"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: postinstall skipifsilent nowait

[UninstallDelete]
; Everything below is ours: the payload plus caches the runtime writes beside it.
; User data (projects, settings, downloaded weights) lives in %LOCALAPPDATA% and
; %APPDATA%\OpenDroneKit and is deliberately left alone.
Type: filesandordirs; Name: "{app}\odk"
Type: filesandordirs; Name: "{app}\runtime"
Type: dirifempty; Name: "{app}"

[Code]
function WebView2Installed(): Boolean;
var
  Version: String;
begin
  Result :=
    (RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')) or
    (RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')) or
    (RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0'));
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and (not WebView2Installed()) and (not WizardSilent()) then
    MsgBox('The Microsoft Edge WebView2 runtime was not found. OpenDroneKit draws its window with it, '
      + 'so the window will not open until it is installed. Windows 11 includes it; on Windows 10 get the '
      + 'Evergreen runtime from https://developer.microsoft.com/microsoft-edge/webview2/',
      mbInformation, MB_OK);
end;
