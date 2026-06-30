# Start CVAT with SAM2 and auto-initialization (PowerShell)
# Uses .env file to automatically load all compose files

Write-Host "🚀 Starting CVAT with SAM2..." -ForegroundColor Green

# Copy Windows-specific .env file
Copy-Item -Path ".env.windows" -Destination ".env" -Force

docker compose up -d --build

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ CVAT is starting up!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Monitor progress:" -ForegroundColor Cyan
    Write-Host "   docker logs nuclio_init -f"
    Write-Host ""
    Write-Host "🌐 Access CVAT at: http://localhost:8080" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "⏱️  SAM2 deployment takes 2-5 minutes on first run" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "❌ Failed to start CVAT" -ForegroundColor Red
    Write-Host "Check Docker is running and try again" -ForegroundColor Yellow
}