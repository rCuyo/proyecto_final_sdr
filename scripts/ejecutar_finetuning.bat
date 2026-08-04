@echo off
setlocal EnableExtensions

if "%~1"=="__RELANZADO__" goto :ENTRENAR

rem ============================================================
rem  Punto de entrada.
rem  Si Windows Terminal esta instalado, abre 3 pestanas:
rem  Entrenamiento + TensorBoard + Monitoreo GPU, automaticamente.
rem  Si no, cae al modo tradicional: una sola ventana de CMD con
rem  el entrenamiento (el usuario puede abrir los otros dos
rem  scripts a mano si quiere).
rem ============================================================
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_DIR=%%~fI"

where wt.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Windows Terminal detectado: abriendo pestanas de Entrenamiento + TensorBoard + GPU...
    wt.exe new-tab --title "Entrenamiento bge-m3" -d "%PROJECT_DIR%" cmd /k call "scripts\ejecutar_finetuning.bat" __RELANZADO__ ; new-tab --title "TensorBoard" -d "%PROJECT_DIR%" cmd /k call "scripts\abrir_tensorboard.bat" ; new-tab --title "Monitoreo GPU" -d "%PROJECT_DIR%" powershell -NoExit -ExecutionPolicy Bypass -File "scripts\monitorear_gpu.ps1"
    if not errorlevel 1 (
        exit /b 0
    )
    echo No se pudo abrir Windows Terminal ^(codigo %ERRORLEVEL%^), se usara una ventana de CMD tradicional...
)

start "Entrenamiento bge-m3" cmd /k call "%~f0" __RELANZADO__
exit /b 0

:ENTRENAR
rem ============================================================
rem  Entrenamiento propiamente dicho (corre en su propia ventana,
rem  ya sea la pestana de Windows Terminal o la ventana de CMD
rem  tradicional abierta arriba).
rem ============================================================
title Fine-Tuning bge-m3 - Entrenamiento

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."

echo ============================================================
echo   FINE-TUNING BAAI/bge-m3 - ENTRENAMIENTO
echo ============================================================
echo Carpeta del proyecto: %CD%
echo.

if exist "venv\Scripts\activate.bat" (
    echo [entorno] Activando entorno virtual: venv
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    echo [entorno] Activando entorno virtual: .venv
    call ".venv\Scripts\activate.bat"
) else if exist "env\Scripts\activate.bat" (
    echo [entorno] Activando entorno virtual: env
    call "env\Scripts\activate.bat"
) else (
    echo [entorno] No se detecto entorno virtual: se usara el Python global ^(py^).
)

echo.
echo Iniciando entrenamiento... esto puede tardar bastante.
echo Log en vivo tambien disponible en: logs\training.log
echo ------------------------------------------------------------
echo.

py scripts\finetuning.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ------------------------------------------------------------
if "%EXIT_CODE%"=="0" (
    echo   ENTRENAMIENTO FINALIZADO CORRECTAMENTE
) else (
    echo   EL ENTRENAMIENTO TERMINO CON ERRORES ^(codigo de salida %EXIT_CODE%^)
    echo   Revisa logs\training.log para mas detalle.
)
echo ------------------------------------------------------------

popd
echo.
pause
endlocal
