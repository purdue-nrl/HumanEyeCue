# Build from the capture directory: python -m PyInstaller --clean app.spec
from pathlib import Path

capture_root = Path(SPECPATH)

a = Analysis(
    [str(capture_root / 'app.py')],
    pathex=[str(capture_root)],
    binaries=[],
    datas=[
        (str(capture_root / 'UI'), 'UI'),
        (str(capture_root / 'icons' / 'icon.ico'), 'icons'),
        (str(capture_root / 'icons' / 'icon.svg'), 'icons'),
    ],
    hiddenimports=['PyQt5.sip'],
    hookspath=[],
    runtime_hooks=[str(capture_root / 'packaging' / 'runtime_paths.py')],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HumanEyeCueCapture',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(capture_root / 'icons' / 'icon.ico'),
    contents_directory='.',
)
distribution = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='HumanEyeCueCapture',
)
