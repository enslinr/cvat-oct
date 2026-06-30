# Stop CVAT (PowerShell)

Write-Host "🛑 Stopping CVAT..." -ForegroundColor Yellow
docker compose down

Write-Host "✅ CVAT stopped" -ForegroundColor Green