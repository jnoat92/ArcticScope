@echo off
REM ============================================
REM Build executable for ArcticScope (PyInstaller)
REM ============================================

REM ---- Your chosen output root (OneDrive path) ----
set ROOT="Your_path\ArcticScope-executable"

echo Cleaning previous build...

REM ---- Delete old dist folder ----
if exist %ROOT%\ArcticScope (
    echo Removing old dist folder: %ROOT%\ArcticScope
    rmdir /S /Q %ROOT%\ArcticScope
)

REM ---- Delete old build folder ----
if exist build\ArcticScope (
    echo Removing old build folder: .\build\ArcticScope
    rmdir /S /Q build\ArcticScope
)

echo.
echo Starting PyInstaller build...

pyinstaller main.py ^
  --onedir ^
  --name ArcticScope ^
  --distpath %ROOT% ^
  --add-data "icons;icons" ^
  --add-data "model;model" ^
  --add-data "landmask;landmask" ^
  --noconfirm ^
  --clean ^
  --hidden-import rasterio.serde ^
  --collect-submodules rasterio ^
  --collect-data rasterio ^
  --hidden-import fiona ^
  --collect-submodules fiona ^
  --collect-data fiona ^
  --collect-binaries fiona ^
  --collect-submodules numpy ^
  --collect-data numpy ^
  --collect-binaries numpy ^
  --collect-submodules scipy ^
  --collect-data scipy ^
  --collect-binaries scipy ^
  --collect-submodules torch ^
  --collect-data torch ^
  --collect-binaries torch

echo.
echo ============================================
echo Build complete. Output is in: %ROOT%\ArcticScope
echo ============================================
echo.
pause
