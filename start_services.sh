#!/bin/bash

# 1. Docker 실행 여부 확인 및 실행
if ! docker info >/dev/null 2>&1; then
    echo "🐳 Docker Desktop 실행 중..."
    open -a Docker
    # Docker가 완전히 켜질 때까지 대기
    while ! docker info >/dev/null 2>&1; do
        echo "⏳ Docker 시작 대기 중..."
        sleep 5
    done
fi

# 2. Docker Compose 인프라 기동
cd "/Users/hamyoungjin/Desktop/AI PJT /MacOS PJT/Setup_AIstudio"
docker-compose up -d

# 3. Auto-Pilot 백그라운드 실행
# 기존 프로세스 종료 후 재실행
pkill -f auto_pilot.py
source .venv/bin/activate
nohup python auto_pilot.py > auto_pilot.log 2>&1 &

echo "🚀 모든 AI 인프라 엔진이 백그라운드에서 기동되었습니다!"
