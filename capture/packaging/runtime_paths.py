"""Resolve the capture app's relative files beside the packaged executable."""
import os
from pathlib import Path
import sys

if getattr(sys, 'frozen', False):
    os.chdir(Path(sys.executable).resolve().parent)
