<#
    Monitoreo en vivo de la GPU (uso, VRAM, temperatura, consumo de energia)
    via nvidia-smi, refrescando cada 2 segundos. Pensado para correr en
    paralelo al entrenamiento (scripts\ejecutar_finetuning.bat/.ps1).
#>

$intervaloSegundos = 2

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "No se encontro 'nvidia-smi' en el PATH. Verifica los drivers de NVIDIA." -ForegroundColor Red
    Read-Host "Presiona Enter para cerrar"
    exit 1
}

$host.UI.RawUI.WindowTitle = "Monitoreo GPU - Fine-Tuning bge-m3"

Write-Host "Monitoreando GPU cada $intervaloSegundos segundos. Ctrl+C para salir." -ForegroundColor Cyan
Start-Sleep -Milliseconds 500

while ($true) {
    $consulta = & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits

    Clear-Host
    Write-Host "============================================================"
    Write-Host "  MONITOREO GPU EN VIVO (se actualiza cada $intervaloSegundos s)"
    Write-Host "============================================================"
    Write-Host "Hora: $(Get-Date -Format 'HH:mm:ss')"
    Write-Host ""

    foreach ($linea in $consulta) {
        $campos = $linea -split ",\s*"
        if ($campos.Count -ge 7) {
            $nombre = $campos[0]
            $usoGpu = $campos[1]
            $vramUsada = $campos[2]
            $vramTotal = $campos[3]
            $temperatura = $campos[4]
            $potenciaActual = $campos[5]
            $potenciaLimite = $campos[6]

            Write-Host "GPU: $nombre"
            Write-Host ("  Uso GPU:      {0,6} %" -f $usoGpu)
            Write-Host ("  VRAM:         {0,6} / {1} MiB" -f $vramUsada, $vramTotal)
            Write-Host ("  Temperatura:  {0,6} C" -f $temperatura)
            Write-Host ("  Consumo:      {0,6} / {1} W" -f $potenciaActual, $potenciaLimite)
            Write-Host ""
        }
    }

    Write-Host "(Ctrl+C para salir)"
    Start-Sleep -Seconds $intervaloSegundos
}
