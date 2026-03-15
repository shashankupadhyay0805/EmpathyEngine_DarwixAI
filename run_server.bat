@echo off
REM Set FFMPEG_PATH to the folder that contains ffmpeg.exe
set "FFMPEG_PATH=C:\Users\shash\Downloads\ffmpeg-8.0.1-essentials_build.zip\ffmpeg-8.0.1-essentials_build\bin"

cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m app.main --serve
