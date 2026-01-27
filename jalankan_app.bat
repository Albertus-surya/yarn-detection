@echo off
setlocal

REM === PINDAH KE FOLDER PROJECT ===
cd /d "%~dp0"

echo =====================================
echo    SETUP ^& RUN PROJECT (SMART CHECK)
echo =====================================

REM ================================
REM 1. Membuat virtual environment
REM ================================
if not exist .venv (
    echo [INFO] Membuat virtual environment baru...
    python -m venv .venv
)

REM ================================
REM 2. Masuk ke virtual environment
REM ================================
echo [INFO] Mengaktifkan virtual environment...
call .venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Gagal aktivasi venv. Cek instalasi Python.
    pause
    exit /b
)

REM ================================
REM 3. SMART INSTALL DEPENDENCIES
REM ================================
set "REQ_FILE=requirements.txt"
set "BAK_FILE=.venv\requirements.bak"
set "NEED_INSTALL=0"

REM Cek apakah requirements.txt ada
if not exist %REQ_FILE% (
    echo [WARNING] File requirements.txt tidak ditemukan!
    goto SKIP_INSTALL
)

REM Cek apakah backup ada. Jika tidak ada, berarti ini install pertama.
if not exist "%BAK_FILE%" (
    set "NEED_INSTALL=1"
    echo [INFO] Instalasi pertama kali terdeteksi.
) else (
    REM Bandingkan file asli dengan backup
    fc /b "%REQ_FILE%" "%BAK_FILE%" >nul
    if errorlevel 1 (
        set "NEED_INSTALL=1"
        echo [INFO] Perubahan pada requirements.txt terdeteksi.
    ) else (
        echo [SKIP] Library sudah terinstall dan up-to-date.
    )
)

REM Proses Install jika diperlukan
if "%NEED_INSTALL%"=="1" (
    echo [INFO] Menginstall/Update library...
    python -m pip install --upgrade pip
    pip install -r %REQ_FILE%
    
    if errorlevel 1 (
        echo [ERROR] Gagal menginstall library! Cek koneksi internet.
        pause
        exit /b
    )
    
    REM Update file backup tanda sukses
    copy /y "%REQ_FILE%" "%BAK_FILE%" >nul
    echo [OK] Instalasi selesai.
)

:SKIP_INSTALL
echo.

REM ================================
REM 4. Cek & Train Model
REM ================================
if not exist models mkdir models

if not exist models\rf_yarn_model_v2.pkl (
    echo [INFO] Model belum ada, training dimulai...
    python train_model_rf.py
) else (
    echo [SKIP] Model sudah ada.
)
echo.

REM ================================
REM 5. Jalankan Aplikasi
REM ================================
echo [INFO] Memulai Aplikasi...

REM --- Pastikan nama file di bawah benar ---
python -m streamlit run app_rf.py

REM Jika script app_rf.py error, pause agar bisa baca errornya
if errorlevel 1 (
    echo.
    echo [ERROR] Aplikasi crash/error!
    pause
)

echo.
echo ===== SELESAI =====
pause
endlocal