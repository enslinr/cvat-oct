#!/bin/bash
# Start CVAT with SAM2 and auto-initialization

echo "🚀 Starting CVAT with SAM2..."

# Detect OS and copy appropriate .env file
# Git Bash on Windows sets OSTYPE to msys or MSYSTEM to MINGW*
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ -n "$MSYSTEM" ]]; then
    echo "Detected Windows environment (Git Bash)"
    cp .env.windows .env
else
    echo "Detected Linux/Mac environment"
    cp .env.linux .env
fi

docker compose up -d

echo ""
echo "✅ CVAT is starting up!"
echo ""
echo "📊 Monitor progress:"
echo "   docker logs nuclio_init -f"
echo ""
echo "🌐 Access CVAT at: http://localhost:8080"
echo ""
echo "⏱️  SAM2 deployment takes 2-5 minutes on first run"
