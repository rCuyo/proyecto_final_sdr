@echo off
setlocal EnableExtensions
title TensorBoard - Fine-Tuning bge-m3

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."

echo ============================================================
echo   TENSORBOARD - Fine-Tuning bge-m3
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

if not exist "runs" (
    echo.
    echo ADVERTENCIA: la carpeta "runs" todavia no existe.
    echo Se crea automaticamente al correr scripts\finetuning.py.
)

echo.
echo Iniciando TensorBoard en una ventana nueva ^(http://localhost:6006^)...
rem Se usa "py -m tensorboard.main" en vez del comando "tensorboard" a secas:
rem el .exe vive en la carpeta Scripts de Python, que no siempre esta en el PATH.
start "TensorBoard - runs" cmd /k py -m tensorboard.main --logdir runs

echo Esperando a que el servidor levante (tarda unos segundos en importar)...
timeout /t 8 /nobreak >nul

echo Abriendo el navegador...
start "" "http://localhost:6006"

popd
echo.
echo Listo. La ventana "TensorBoard - runs" debe quedar abierta mientras la uses.
echo Cerrala (o Ctrl+C ahi dentro) para detener el servidor.
pause
endlocal
