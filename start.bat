@echo off
rem ============================================================
rem  DeepResearch Agent - one-click launcher
rem  Double-click this file, or run from a terminal:
rem    start.bat [-NoBrowser] [-NoBuild]
rem  It calls scripts\start.ps1 which starts the full stack
rem  (postgres / redis / searxng / api / worker / dispatcher / web)
rem  via Docker Compose and opens http://localhost:5174.
rem ============================================================
chcp 65001 >nul
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo [FAIL] 启动失败（退出码 %EXIT_CODE%），请根据上方错误信息处理。
    pause
    exit /b %EXIT_CODE%
)
echo [OK] 启动流程已结束，可关闭本窗口。
pause
endlocal
