@echo off
REM Run script for Space Robotics and AI Docker environment (Windows)

REM Check if docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not installed
    pause
    exit /b 1
)

REM Reminder for X Server
timeout /t 3 >nul

REM Try to start with docker-compose
echo Starting container with docker-compose...
docker-compose up -d 2>nul
if not errorlevel 1 goto :compose_success

REM Try docker compose v2
echo Trying docker compose v2...
docker compose up -d 2>nul
if not errorlevel 1 goto :compose_v2_success

REM Fallback to docker run
echo Using docker run...
docker run -it --rm --name srai-workspace -e DISPLAY=host.docker.internal:0 -e QT_X11_NO_MITSHM=1 -v "%cd%/assignments:/workspace/src:rw" space-robotics-ai:humble
goto :end

:compose_success
echo Container started successfully!
docker-compose exec space-robotics-dev bash
goto :end

:compose_v2_success
echo Container started successfully!
docker compose exec space-robotics-dev bash
goto :end

:end
echo Exited container.
pause
