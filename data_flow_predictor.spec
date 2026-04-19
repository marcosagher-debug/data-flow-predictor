# data_flow_predictor.spec
# Compilar con: pyinstaller data_flow_predictor.spec

import sys
import os
from pathlib import Path
import streamlit

block_cipher = None

# ── Rutas críticas ───────────────────────────────────────────────────────────
STREAMLIT_PATH = Path(streamlit.__file__).parent
APP_DIR = Path(__file__).parent

# ── Recolección de datos de Streamlit ────────────────────────────────────────
streamlit_datas = [
    (str(STREAMLIT_PATH / "static"),    "streamlit/static"),
    (str(STREAMLIT_PATH / "runtime"),   "streamlit/runtime"),
    (str(STREAMLIT_PATH / "components"),"streamlit/components"),
]

# ── Datos del proyecto ────────────────────────────────────────────────────────
project_datas = [
    (str(APP_DIR / "data"),   "data"),
    (str(APP_DIR / "assets"), "assets"),
]

all_datas = streamlit_datas + project_datas

a = Analysis(
    ['launcher.py'],           # launcher abre streamlit run main_app.py
    pathex=[str(APP_DIR)],
    binaries=[],
    datas=all_datas,
    hiddenimports=[
        'streamlit',
        'streamlit.runtime.scriptrunner',
        'streamlit.web.cli',
        'tensorflow',
        'sklearn',
        'sklearn.preprocessing._label',
        'statsmodels',
        'statsmodels.formula.api',
        'openpyxl',
        'reportlab',
        'joblib',
        'seaborn',
        'matplotlib',
        'pandas',
        'numpy',
        'scipy',
        'scipy.stats',
        'altair',
        'pyarrow',
        'packaging',
        'click',
        'watchdog',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DataFlowPredictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # Sin ventana de consola (windowed)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/imp_icon.ico',   # Coloca tu ícono aquí
)
