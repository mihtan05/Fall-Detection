@echo off
title AI Real-Time Fall Detection System
echo ============================================================
echo   DANG KHOI DONG HE THONG AI PHAT HIEN TE NGA (YOLOv11)
echo ============================================================
echo.
cd /d "%~dp0"
".venv\Scripts\python.exe" run_webcam.py
if errorlevel 1 (
    echo.
    echo [!] Co loi xay ra hoac chuong trinh da dung.
    pause
)
