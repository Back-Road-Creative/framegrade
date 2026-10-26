@echo off
rem Framegrade launcher: starts the GPU host worker and the console with the
rem bundled Python, then opens the console in the browser. No terminal needed;
rem the Start-menu shortcut runs this minimised.
rem
rem Keep this file pure ASCII (cmd reads it in the console code page).
setlocal
set "HERE=%~dp0"
set "PY=%HERE%python\python.exe"

rem The data folder is chosen once in the installer, which writes the path to
rem data-folder.txt beside this file. Without the file, fall back to a folder
rem in the user profile.
set "GMS_DATA_HOME=%USERPROFILE%\Framegrade"
if exist "%HERE%data-folder.txt" set /p GMS_DATA_HOME=<"%HERE%data-folder.txt"

rem Everything the app writes lives under that one folder.
set "GMS_CACHE_HOME=%GMS_DATA_HOME%\cache"
set "GMS_CONFIG_HOME=%GMS_DATA_HOME%\config"
set "SEO_MANAGER_DATA_HOME=%GMS_DATA_HOME%\waypost"
set "GMS_HOST_SPOOL_DIR=%GMS_DATA_HOME%\host-spool"
set "PYTHONUTF8=1"
for %%D in ("%GMS_DATA_HOME%" "%GMS_CACHE_HOME%" "%GMS_CONFIG_HOME%" "%SEO_MANAGER_DATA_HOME%" "%GMS_HOST_SPOOL_DIR%") do if not exist "%%~D" mkdir "%%~D"

rem The CUDA and cuDNN runtime DLLs ship as pip wheels under site-packages\nvidia.
for /d %%D in ("%HERE%python\Lib\site-packages\nvidia\*") do if exist "%%D\bin" set "PATH=%%D\bin;%PATH%"

start "Framegrade host worker" /min "%PY%" -m gomoveshift.hostworker
start "Framegrade console" /min "%PY%" -m gomoveshift console

rem Give the console a few seconds to bind its port before the browser asks.
timeout /t 6 /nobreak >nul
start "" http://127.0.0.1:8787
endlocal
