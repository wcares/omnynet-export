@echo off
rem Build every Windows processor of this repo with the native MSVC toolchain (x86_64-pc-windows-msvc). Run from anywhere; output: %CARGO_TARGET_DIR% (default <this dir>\target)\x86_64-pc-windows-msvc\release\<name>-processor.exe
rem Needs: Visual Studio Build Tools (C++ workload), rustup with the stable toolchain and the x86_64-pc-windows-msvc target.
rem semantic-processor is not built: the semantic job runs in-process in omny-compute on every OS.
rem Usage: build-windows.cmd            builds all nine
rem        build-windows.cmd ocr-processor clip-processor     builds only the named packages
setlocal enableextensions enabledelayedexpansion
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH ( echo ERROR: vswhere found no Visual Studio install & exit /b 1 )
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set CC=cl
set CXX=cl
set CARGO_INCREMENTAL=0
cd /d "%~dp0" || exit /b 1
set "PKGS=%*"
if "%PKGS%"=="" set "PKGS=ocr-processor paddleocr-processor doc-layout-processor table-structure-processor clip-processor clip-text-processor dino-processor sam2-processor whisper-processor"
for %%p in (%PKGS%) do (
  echo BUILDING %%p
  cargo +stable build --release --target x86_64-pc-windows-msvc -p %%p || exit /b 2
)
if defined CARGO_TARGET_DIR (set "OUT=%CARGO_TARGET_DIR%\x86_64-pc-windows-msvc\release") else (set "OUT=%~dp0target\x86_64-pc-windows-msvc\release")
for %%p in (%PKGS%) do (
  for /f "usebackq tokens=*" %%h in (`powershell -NoProfile -Command "(Get-FileHash '!OUT!\%%p.exe' -Algorithm SHA256).Hash.ToLower()"`) do echo %%p.exe sha256 %%h
)
echo PROCESSOR BUILD OK