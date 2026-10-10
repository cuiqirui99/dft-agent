#ifndef AppVersion
  #define AppVersion "0.4.2"
#endif

[Setup]
AppId=DFTAgent
AppName=DFT Agent
AppVersion={#AppVersion}
AppPublisher=Qirui Cui
AppPublisherURL=https://github.com/cuiqirui99/dft-agent
DefaultDirName={localappdata}\Programs\DFT Agent
DefaultGroupName=DFT Agent
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installers
OutputBaseFilename=DFT-Agent-{#AppVersion}-Windows-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\DFT Agent.exe
CloseApplications=yes
LicenseFile=..\LICENSE

[Files]
Source: "..\dist\DFT Agent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\dist\MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: NeedsWebView2

[Icons]
Name: "{group}\DFT Agent"; Filename: "{app}\DFT Agent.exe"
Name: "{autodesktop}\DFT Agent"; Filename: "{app}\DFT Agent.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "Installing Microsoft WebView2 (internet required)..."; Flags: waituntilterminated; Check: NeedsWebView2
Filename: "{app}\DFT Agent.exe"; Description: "Open DFT Agent"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsWebView2: Boolean;
var
  Version: String;
begin
  Result := not ((RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')) or
    (RegQueryStringValue(HKLM32, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')));
end;
