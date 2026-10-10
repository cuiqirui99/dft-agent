# Desktop builds

`0.4.2` is an unpublished local preview. The packaging targets are Apple
Silicon macOS and Windows x64. Python and the scientific/model SDKs are
bundled; VASP, POTCAR files, cluster time, API credentials and model credits
are not. This page describes how to build and check a package, not a claim
that every platform or external service has passed acceptance.

## Build from a checkout

Use a separate virtual environment on the target operating system and architecture:

```bash
python -m pip install '.[desktop,dev]' 'pyinstaller>=6.15,<7' 'ase>=3.23,<4'
python -m PyInstaller --noconfirm packaging/desktop.spec
```

On macOS, this produces `dist/DFT Agent.app`. Create a disk image after the
app has passed its checks:

```bash
mkdir -p dist/dmg-preview
cp -R 'dist/DFT Agent.app' dist/dmg-preview/
ln -s /Applications dist/dmg-preview/Applications
hdiutil create -volname 'DFT Agent' -srcfolder dist/dmg-preview -ov -format UDZO dist/DFT-Agent-0.4.2-macOS-arm64.dmg
```

On Windows, the app folder is `dist/DFT Agent`. The
[Windows packaging workflow](../.github/workflows/desktop.yml) downloads and
checks Microsoft's WebView2 installer, invokes Inno Setup with
[windows.iss](../packaging/windows.iss), and checks a fresh installation.
The resulting installer goes into `dist/installers/`. Run this on Windows;
a macOS build is not a Windows installation test.

## Check the installed copy

For example, after copying the Mac app into Applications:

```bash
python packaging/check_installed.py '/Applications/DFT Agent.app/Contents/MacOS/DFT Agent' evidence/macos
```

The same command accepts the installed Windows `DFT Agent.exe` path. It
creates isolated settings and runs, checks bundled resources, sample copies,
local input preparation and cluster attachment, then starts the packaged
local server. It records `bundle-check.json` and `server.log`. This requires
no model account or cluster and does not send a job.

Also open the installed app in a fresh user workspace and verify sample
viewing, manual preparation, language switching, configuration editing,
file selection and downloads. Keep the installer hash, commands, reports
and screenshots with the acceptance record. Passing unit tests or the
local-server check alone does not verify the native window or every user action.

Live SSH/VASP calculations and live model calls require separate evidence.
The [recorded Claude checks](validation-0.4.1.md) used the actual SDK with
simulated HTTP responses; no live Claude API request was made. Code signing,
notarization, Windows installer execution and interactive checks should only
be marked complete when their own records exist.

Publish each completed version under a new tag and GitHub Release. Keep
previous release records and assets unchanged; local preview artifacts do
not establish that a release has been published.
