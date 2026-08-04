<#
    Ejecuta el entrenamiento de fine-tuning (scripts\finetuning.py).
    Hace exactamente lo mismo que ejecutar_finetuning.bat, en PowerShell:
    activa el entorno virtual si existe, se posiciona en la carpeta del
    proyecto, corre el entrenamiento y muestra si termino bien o con error.

    Si Windows Terminal esta instalado, abre 3 pestanas (Entrenamiento +
    TensorBoard + Monitoreo GPU) automaticamente. Si no, abre una sola
    ventana de PowerShell con el entrenamiento.
#>

param(
    [switch]$Relanzado
)

$carpetaScript = Split-Path -Parent $MyInvocation.MyCommand.Path
$carpetaProyecto = Split-Path -Parent $carpetaScript

if (-not $Relanzado) {
    $wt = Get-Command wt.exe -ErrorAction SilentlyContinue
    if ($wt) {
        Write-Host "Windows Terminal detectado: abriendo pestanas de Entrenamiento + TensorBoard + GPU..."
        try {
            $rutaPropia = $MyInvocation.MyCommand.Path
            $rutaMonitoreo = Join-Path $carpetaScript "monitorear_gpu.ps1"

            & $wt.Source new-tab --title "Entrenamiento bge-m3" -d "$carpetaProyecto" powershell -NoExit -ExecutionPolicy Bypass -File "$rutaPropia" -Relanzado `; `
                new-tab --title "TensorBoard" -d "$carpetaProyecto" cmd /k call "scripts\abrir_tensorboard.bat" `; `
                new-tab --title "Monitoreo GPU" -d "$carpetaProyecto" powershell -NoExit -ExecutionPolicy Bypass -File "$rutaMonitoreo"

            if ($LASTEXITCODE -eq 0) {
                exit 0
            }
            Write-Host "No se pudo abrir Windows Terminal (codigo $LASTEXITCODE), se usara una ventana tradicional..." -ForegroundColor Yellow
        } catch {
            Write-Host "No se pudo abrir Windows Terminal ($_), se usara una ventana tradicional..." -ForegroundColor Yellow
        }
    }

    # Modo tradicional: relanzarse en una ventana nueva de PowerShell.
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$($MyInvocation.MyCommand.Path)`"", "-Relanzado"
    )
    exit 0
}

# ============================================================
#  Entrenamiento propiamente dicho.
# ============================================================
$host.UI.RawUI.WindowTitle = "Fine-Tuning bge-m3 - Entrenamiento"
Set-Location $carpetaProyecto

Write-Host "============================================================"
Write-Host "  FINE-TUNING BAAI/bge-m3 - ENTRENAMIENTO"
Write-Host "============================================================"
Write-Host "Carpeta del proyecto: $carpetaProyecto"
Write-Host ""

$activado = $false
foreach ($nombre in @("venv", ".venv", "env")) {
    $rutaActivar = Join-Path $carpetaProyecto "$nombre\Scripts\Activate.ps1"
    if (Test-Path $rutaActivar) {
        Write-Host "[entorno] Activando entorno virtual: $nombre"
        & $rutaActivar
        $activado = $true
        break
    }
}
if (-not $activado) {
    Write-Host "[entorno] No se detecto entorno virtual: se usara el Python global (py)."
}

Write-Host ""
Write-Host "Iniciando entrenamiento... esto puede tardar bastante."
Write-Host "Log en vivo tambien disponible en: logs\training.log"
Write-Host "------------------------------------------------------------"
Write-Host ""

py scripts\finetuning.py
$codigoSalida = $LASTEXITCODE

Write-Host ""
Write-Host "------------------------------------------------------------"
if ($codigoSalida -eq 0) {
    Write-Host "  ENTRENAMIENTO FINALIZADO CORRECTAMENTE" -ForegroundColor Green
} else {
    Write-Host "  EL ENTRENAMIENTO TERMINO CON ERRORES (codigo de salida $codigoSalida)" -ForegroundColor Red
    Write-Host "  Revisa logs\training.log para mas detalle." -ForegroundColor Red
}
Write-Host "------------------------------------------------------------"

Write-Host ""
Read-Host "Presiona Enter para cerrar"
